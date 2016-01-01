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
        
        struct socket_info
        {
            int m_socket;
            struct sockaddr_in m_socket_addr;
        };
        
        enum result
        {
            RESULT_OK,
            RESULT_FAIL_TO_CREATE_SOCKET,
            RESULT_FAIL_SET_NONBLOCK,
            RESULT_FAIL_SET_ASYNC,
            RESULT_FAIL_TO_CONNECTION,
        };
        
        socket_messages_server(int port)
        {
            //default: error socket
            m_server.m_socket = -1;
            //info
            m_port = port;
            m_run  = true;
            //clean list
            m_client_list.clear();
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
                result nb_async_ret = set_nonblocking_async(m_server.m_socket);
                //test
                if( nb_async_ret != RESULT_OK )
                {
                    m_error = nb_async_ret;
                    return;
                }
                //try to connect
                if (bind(m_server.m_socket, (struct sockaddr *) &m_server.m_socket_addr, sizeof(m_server.m_socket_addr)) < 0)
                {
                    m_error = RESULT_FAIL_TO_CONNECTION;
                    m_run=false;
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
            m_messages.push(str);
        }
        
        
        
    protected:
        
        //close connection and stop thread
        void close()
        {
            if(m_run)
            {
                m_run = false;
                m_thread.join();
                close_server_socket();
            }
        }
        
        
        //set nonbloking
        static result set_nonblocking_async(int socket)
        {
            if( fcntl(socket, F_SETFL, fcntl(socket, F_GETFL, 0) | O_NONBLOCK) == -1 )
            {
                return RESULT_FAIL_SET_NONBLOCK;
            }
            
            if( fcntl(socket, F_SETFL, fcntl(socket, F_GETFL, 0) | O_ASYNC) == -1 )
            {
                return RESULT_FAIL_SET_ASYNC;
            }
            
            return RESULT_OK;
        }
        //accept now sockets
        void accept_client()
        {
            //data of new socket
            socket_info new_socket;
            //ref
            struct sockaddr* ref_socket_addr = (struct sockaddr*)&new_socket.m_socket_addr;
            socklen_t size_addr              = sizeof(struct sockaddr_in);
            //accept
            new_socket.m_socket = accept(m_server.m_socket, ref_socket_addr, &size_addr);
            //test
            if( new_socket.m_socket >= 0 )
            {
                m_client_list.push(new_socket);
            }
        }
        //send message
        void send_messages()
        {
            while(m_messages.size())
            {
                //message
                const std::string& first = m_messages[0];
                //to all clients
                for(auto& s_info : m_client_list)
                {
                    ::write(s_info.m_socket, (void*)first.c_str(), first.size());
                }
                m_messages.remove_first();
            }
        }
        //close all connection
        void close_server_socket()
        {
            if(m_server.m_socket!=-1)
            {
                for(auto& s_info : m_client_list)
                {
                    ::shutdown(s_info.m_socket, SHUT_RDWR);
                    ::close(s_info.m_socket);
                }
                ::close(m_server.m_socket);
            }
            m_client_list.clear();
            m_server.m_socket = -1;
        }
        //soket info
        atomic_vector < socket_info > m_client_list;
        socket_info                   m_server;
        atomic_vector < std::string > m_messages;
        //thread info
        std::thread              m_thread;
        std::atomic< bool >      m_run  { 0 };
        std::atomic< int >       m_error{ 0 };
        int                      m_port { 0 };
        
    };
    
}
}
