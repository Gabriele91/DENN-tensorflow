//
//  socket_messages_server.h
//  DifferentialEvolutionOp
//
//  Created by Gabriele Di Bari on 22/11/16.
//  Copyright © 2016 Gabriele. All rights reserved.
//
#pragma once
#include <thread>
#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>

namespace tensorflow
{
namespace debug
{
    class socket_messages_server
    {
    public:
        
        
        struct socket_info
        {
            //fields
            int m_socket{ -1 };
            struct sockaddr_in m_socket_addr;
            
            //standard init
            socket_info(){}
            
            //init info
            socket_info(int in_socket,const struct sockaddr_in& in_socket_addr)
            :m_socket(in_socket)
            ,m_socket_addr(in_socket_addr)
            {
            }
            
        };

        template < class T >
        class atomic_vector
        {
        public:
            
            using iterator = typename std::vector< T >::const_iterator;
            using type     = T;
            
            void push(const T& value)
            {
                m_mutex.lock();
                m_vector.push_back(value);
                m_mutex.unlock();
            }
            
            void remove_first()
            {
                m_mutex.lock();
                auto elm = m_vector.erase(m_vector.begin());
                m_mutex.unlock();
            }
            
            const T& operator[](size_t i) const
            {
                return m_vector[i];
            }
            
            size_t size() const
            {
                return m_vector.size();
            }
            
            void clear()
            {
                m_vector.clear();
            }
            
            iterator begin() const
            {
                return m_vector.begin();
            }
            
            iterator end() const
            {
                return m_vector.end();
            }
            
        protected:
            std::mutex       m_mutex;
            std::vector< T > m_vector;
        };
        
        enum message_type 
        {
            MSG_INT,
            MSG_FLOAT,
            MSG_DOUBLE,
            MSG_STRING
        };

        using message_raw = std::vector< unsigned char >;

        enum result
        {
            RESULT_OK,
            RESULT_FAIL_TO_CREATE_SOCKET,
            RESULT_FAIL_SET_NONBLOCK,
            RESULT_FAIL_SET_ASYNC,
            RESULT_FAIL_TO_CONNECTION,
            RESULT_FAIL_TO_LISTEN
        };
        
        socket_messages_server(int port)
        {
            //default: error socket
            m_server.m_socket = -1;
            //info
            m_port = port;
            m_run  = true;
            //clean
            m_client = socket_info();
            //start
            m_thread = std::thread(
            [this]()
            {
                //create addres connection
                m_server.m_socket_addr.sin_family = AF_INET;
                m_server.m_socket_addr.sin_addr.s_addr = INADDR_ANY;
                m_server.m_socket_addr.sin_port = htons(m_port);
                //create socket
                m_server.m_socket = socket(AF_INET, SOCK_STREAM, 0);
                //test
                if (m_server.m_socket < 0)
                {
                    m_error = RESULT_FAIL_TO_CREATE_SOCKET;
                    m_run=false;
                    return;
                }
                //set nonblocking and async
                result nb_ret = set_nonblocking(m_server.m_socket);
                //test
                if( nb_ret != RESULT_OK )
                {
                    m_error = nb_ret;
                    return;
                }
                //try to connect
                if (bind(m_server.m_socket, (struct sockaddr *) &m_server.m_socket_addr, sizeof(m_server.m_socket_addr)) < 0)
                {
                    m_error = RESULT_FAIL_TO_CONNECTION;
                    m_run = false;
                    return;
                }
                //enale listen
                if(::listen(m_server.m_socket,1) < 0)
                {
                    m_error = RESULT_FAIL_TO_LISTEN;
                    m_run = false;
                    return;
                }
                
                //run
                while(m_run)
                {
                    //add new connection(s)
                    accept_client();
                    //send messages
                    send_messages();
                }
            });
        }
        
        virtual ~socket_messages_server()
        {
            close();
        }
        
        void write(const std::string& str)
        {
            //alloc
            message_raw msg(
                sizeof(unsigned int) // type
              + sizeof(unsigned int) // len
              + str.size() 
            );
            //values 
            unsigned int type = MSG_STRING;
            unsigned int len  = (unsigned int)str.length();
            //write
            std::memcpy(&msg[0                     ],&type,       sizeof(unsigned int));
            std::memcpy(&msg[sizeof(unsigned int)  ],&len,        sizeof(unsigned int));
            std::memcpy(&msg[sizeof(unsigned int)*2],str.c_str(), str.length());
            //add into queue
            m_messages.push(msg);
        }
        
        void write(int i)
        {
            //alloc
            message_raw msg(
                sizeof(unsigned int) // type
              + sizeof(int)          // integer
            );
            //values 
            unsigned int type = MSG_INT;
            //write
            std::memcpy(&msg[0                   ],&type,sizeof(unsigned int));
            std::memcpy(&msg[sizeof(unsigned int)],&i,   sizeof(int));
            //add into queue
            m_messages.push(msg);
        }
        
        void write(float f)
        {
            //alloc
            message_raw msg(
                sizeof(unsigned int) // type
              + sizeof(float)        // float
            );
            //values 
            unsigned int type = MSG_FLOAT;
            //write
            std::memcpy(&msg[0                   ],&type,sizeof(unsigned int));
            std::memcpy(&msg[sizeof(unsigned int)],&f,   sizeof(float));
            //add into queue
            m_messages.push(msg);
        }

        void write(double d)
        {
            //alloc
            message_raw msg(
                sizeof(unsigned int) // type
              + sizeof(double)       // double
            );
            //values 
            unsigned int type = MSG_DOUBLE;
            //write
            std::memcpy(&msg[0                   ],&type,sizeof(unsigned int));
            std::memcpy(&msg[sizeof(unsigned int)],&d,   sizeof(double));
            //add into queue
            m_messages.push(msg);
        }

    protected:

        //close connection and stop thread
        void close()
        {
            if(m_run)
            {
                //stop thrad
                join();
                //close socket
                close_server_socket();
            }
            //stop thread anyway
            else join();
        }
        
        //join thread
        void join()
        {
            //disable loop
            m_run = false;
            //stop thread
            if(m_thread.joinable())
            {
                //join
                m_thread.join();
            }
        }
        
        //set nonbloking
        static result set_nonblocking(int socket)
        {
            if( fcntl(socket, F_SETFL, fcntl(socket, F_GETFL, 0) | O_NONBLOCK) == -1 )
            {
                return RESULT_FAIL_SET_NONBLOCK;
            }
            
            return RESULT_OK;
        }

        //set async
        static result set_async(int socket)
        {
            if( fcntl(socket, F_SETFL, fcntl(socket, F_GETFL, 0) | O_ASYNC) == -1 )
            {
                return RESULT_FAIL_SET_ASYNC;
            }
            
            return RESULT_OK;
        }

        //test if a tcp socket is live
        static bool keepalive(int socket)
        {
            //value
            int is_live = false;
            //size of value
            socklen_t sizeo_of_is_live = sizeof(is_live);
            //get result
            if(getsockopt(socket, SOL_SOCKET, SO_KEEPALIVE, &is_live, &sizeo_of_is_live) < 0)
            {
                return false;
            }
            //return
            return sizeo_of_is_live != 0;
        }
        
        //accept now sockets
        void accept_client()
        {
            if(m_client.m_socket < 0) //|| !keepalive(m_client.m_socket))
            {
                //close stream
                if(m_client.m_socket >= 0) ::close(m_client.m_socket);
                //init
                m_client = socket_info();
                //ref
                struct sockaddr* ref_socket_addr = (struct sockaddr*)&m_client.m_socket_addr;
                socklen_t size_addr              = sizeof(struct sockaddr_in);
                //accept
                m_client.m_socket = accept(m_server.m_socket, ref_socket_addr, &size_addr);
            }
        }
        
        //send message
        void send_messages()
        {
            if(m_client.m_socket >= 0)
            {
                for(size_t i=0; i!=m_messages.size(); ++i)
                {
                    //message
                    const message_raw& data = m_messages[i];
                    //to client
                    ::write(m_client.m_socket, (void*)data.data(), data.size());
                }
            }
            m_messages.clear();
        }
        
        //close all connection
        void close_server_socket()
        {
            if(m_client.m_socket >= 0)
            {
                ::shutdown(m_client.m_socket, SHUT_RDWR);
                ::close(m_client.m_socket);
            }
            if(m_server.m_socket >= 0)
            {
                ::close(m_server.m_socket);
            }
            //clean
            m_server = socket_info();
            m_client = socket_info();
        }
        //soket info
        socket_info m_client;
        socket_info m_server;
        //pessage list
        atomic_vector < message_raw > m_messages;
        //thread info
        std::thread              m_thread;
        std::atomic< bool >      m_run  { 0 };
        std::atomic< int >       m_error{ 0 };
        int                      m_port { 0 };
        
    };
    
}
}
