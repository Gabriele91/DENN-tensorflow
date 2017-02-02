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
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

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
            
            void safe_copy(T& in,int i)
            {
                m_mutex.lock();
                in = m_vector[i];
                m_mutex.unlock();
            }

            #if 0 
            const T& operator[](size_t i) const
            {
                return m_vector[i];
            }
            #endif
            
            size_t size() const
            {
                return m_vector.size();
            }
            
            void clear()
            {
                m_mutex.lock();
                m_vector.clear();
                m_mutex.unlock();
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
            MSG_STRING,
            MSG_CLOSE_CONNECTION
        };

        using message_raw = std::vector< unsigned char >;

        enum result
        {
            RESULT_OK,
            RESULT_FAIL_TO_CREATE_SOCKET,
            RESULT_FAIL_TO_ENABLE_REUSEADDRS,
            RESULT_FAIL_SET_NONBLOCK,
            RESULT_FAIL_SET_ASYNC,
            RESULT_FAIL_TO_CONNECTION,
            RESULT_FAIL_TO_LISTEN
        };
        
        socket_messages_server(int port=8484)
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
                m_server.m_socket_addr.sin_addr.s_addr = htonl (INADDR_ANY);
                m_server.m_socket_addr.sin_port = htons(m_port);
                //create socket
                m_server.m_socket = socket( PF_INET, SOCK_STREAM, IPPROTO_TCP );
                //test
                if (m_server.m_socket < 0)
                {
                    m_error = RESULT_FAIL_TO_CREATE_SOCKET;
                    m_run=false;
                    return;
                }
                //disable linger
                set_linger(m_server.m_socket,0,0);
                //enable reuse of addrs 
                if(!set_reuse_addrs(m_server.m_socket))
                {
                    m_error = RESULT_FAIL_TO_ENABLE_REUSEADDRS;
                    m_run=false;
                    return;
                }
                //set nonblocking
                result nb_ret = set_nonblocking(m_server.m_socket);
                //test
                if( nb_ret != RESULT_OK )
                {
                    m_error = nb_ret;
                    return;
                }
                //try to bind
                result ntryb_ret = n_try_bind(m_server, 10, 100);
                //test
                if( ntryb_ret != RESULT_OK )
                {
                    m_error = ntryb_ret;
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
                    //get new connection
                    accept_client();
#if 1
                    //send messages
                    send_messages();
                    //read messages 
                    read_messages();
#endif
                }
            });
        }
        
        virtual ~socket_messages_server()
        {
            close();
        }
        
        void write(const std::string& str) const
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
            m_send_msg.push(msg);
        }
        
        void write(int i) const
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
            m_send_msg.push(msg);
        }
        
        void write(float f) const
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
            m_send_msg.push(msg);
        }

        void write(double d) const
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
            m_send_msg.push(msg);
        }

        size_t get_n_recv_mgs() const
        {
            return m_recv_msg.size();
        }

        //message temp data
        message_raw pop_recv_msg()
        {
            message_raw data;
            m_recv_msg.safe_copy(data,0);
            m_recv_msg.remove_first();
            return data;
        }

    protected:

        //close connection
        void send_close_connection_immediately()
        {
            //values 
            unsigned int type = MSG_CLOSE_CONNECTION;
            //to client
            ::send(m_client.m_socket, (void*)&type, sizeof(unsigned int), 0);
        }

        //close connection and stop thread
        void close()
        {
            //stop thread
            join();
            //close socket
            close_server_socket();
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
        
        //set linux linger
        static bool set_linger(int socket,int onoff,int secs_to_linger)
        {
            //only on linux
            #if defined( __linux__ )
            //struct
            struct
            {
                int l_onoff;    /* linger active */
                int l_linger;   /* how many seconds to linger for */
            }
            linger;
            //disable
            linger.l_onoff = onoff;
            linger.l_linger = secs_to_linger;
            //set
            if(setsockopt(socket, SOL_SOCKET, SO_LINGER, (int*)&linger, sizeof(linger)) < 0)
            {
                return false;
            }
            #endif
            return true;
        }
        
        //try to bind
        static result n_try_bind(const socket_info& socket,const short n_test,const useconds_t ms_time_to_try)
        {
            //do connect
            for(short i_test=1;  i_test != (n_test+1); ++i_test)
            {
                int ret = bind(socket.m_socket, (struct sockaddr *)
                               &socket.m_socket_addr,
                               sizeof(socket.m_socket_addr));
                //ok
                if(ret >= 0)
                {
                    return  RESULT_OK;
                }
                //wait
                usleep(ms_time_to_try * 1000);
            }
            
            return RESULT_FAIL_TO_CONNECTION;
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

        //set bloking
        static result set_blocking(int socket)
        {
            if( fcntl(socket, F_SETFL, fcntl(socket, F_GETFL, 0)  & (~O_NONBLOCK)) == -1 )
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
            int is_live = 0;
            //size of value
            socklen_t sizeo_of_is_live = sizeof(is_live);
            //get result
            if(getsockopt(socket, SOL_SOCKET, SO_KEEPALIVE, &is_live, &sizeo_of_is_live) < 0)
            {
                return false;
            }
            //return
            return is_live != 0;
        }
        
        // enable re-use addrs 
        static bool set_reuse_addrs(int sockfd)
        {
            int optval = 1;
            return setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR,  &optval, sizeof(optval)) == 0;
        }

        // enable TCP keepalive on the socket
        static bool set_tcp_keepalive(int sockfd)
        {
            int optval = 1;
            return setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &optval, sizeof(optval)) == 0;
        }

        /**
        *  Set time of keep is live
        *  @param keepcnt, The time (in seconds) the connection needs to remain idle before TCP starts sending keepalive probes.
        *  @param keepidle, The maximum number of keepalive probes TCP should send before dropping the connection.      
        *  @param keepintvl, The time (in seconds) between individual keepalive probes.
        **/
        static int set_tcp_keepalive_cfg(int sockfd, int keepcnt, int keepidle,int keepintvl)
        {
            int rc;
            #ifdef __APPLE__
                        //set the keepalive option
                        int seconds = keepcnt + keepidle*keepintvl;
                        rc = setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPALIVE, &seconds, sizeof(seconds));
                        if (rc != 0) return rc;
            #else
                        //set the keepalive options
                        rc = setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
                        if (rc != 0) return rc;

                        rc = setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
                        if (rc != 0) return rc;

                        rc = setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
                        if (rc != 0) return rc;
            #endif
            return 0;
        }

        //safe non-blocking accept
        static bool timeout_accept(int server_socket,int secs,int usecs,socket_info& client_out)
        {
            //alloc struct
            fd_set readfds;
            //to zero
            FD_ZERO(&readfds);
            //init read set
            FD_SET(server_socket,&readfds);
            //init timeout
            timeval select_timeout;
            select_timeout.tv_sec = secs;
            select_timeout.tv_usec = usecs; 
            //try to accept 
            if (select(server_socket+1,&readfds,nullptr,nullptr,&select_timeout) >= 0)
            {
                if (FD_ISSET(server_socket, &readfds))
                {
                    //ref
                    struct sockaddr* ref_socket_addr = (struct sockaddr*)&client_out.m_socket_addr;
                    socklen_t        size_addr       = sizeof(struct sockaddr_in);
                    //accept
                    client_out.m_socket = accept(server_socket, ref_socket_addr, &size_addr);
                    //success
                    return client_out.m_socket >= 0;
                }
            }
            return false;
        }

        //accept now sockets
        void accept_client()
        {
            if(m_client.m_socket >=0 && !keepalive(m_client.m_socket))
            {
                //close socket
                ::close(m_client.m_socket);
                //clean info
                m_client = socket_info();
            }

            if(m_client.m_socket < 0)
            {
                //close stream
                if(m_client.m_socket >= 0) ::close(m_client.m_socket);
                //init
                m_client = socket_info();
                //enable keepalive
                if(timeout_accept(m_server.m_socket, 2, 0, m_client))
                { 
                    set_nonblocking(m_client.m_socket);
                    set_tcp_keepalive(m_client.m_socket);
                    set_tcp_keepalive_cfg(
                          m_client.m_socket
                        , 1 //time to wait (in seconds)
                        , 0  //n-ack to determinete if is live
                        , 0  //time between ack (in seconds)
                    );
                }
            }
        }
        
        //send message
        void send_messages()
        {
            //send
            if((m_client.m_socket >= 0) && (m_send_msg.size()!= 0))
            {
                while(m_send_msg.size())
                {
                    //message temp data
                    message_raw data;
                    //copy
                    m_send_msg.safe_copy(data, 0);
                    //remove
                    m_send_msg.remove_first();
                    //to client
                    if(data.size())
                    {
                        if(::send(m_client.m_socket, (void*)data.data(), data.size(), 0) < 0)
                        {
                            return;
                        } 
                    }
                }
            }
        }

        //send message
        void read_messages()
        {
            //read
            if(m_client.m_socket >= 0)
            {
                //message temp data
                message_raw data; 
                //type size
                const msg_type_size = sizeof(unsigned int);
                //alloc type
                data.resize(msg_type_size);
                //try to recv 
                int ret = ::recv(m_client.m_socket, (void*)data.data(), msg_type_size, 0);
                //read type
                if(ret > 0)
                {
                    //get type
                    unsigned int type = *((unsigned int*)(data.data()));
                    //read by type 
                    switch(type)
                    {
                        case MSG_INT:
                            data.resize(msg_type_size+sizeof(int));   
                            ::recv(m_client.m_socket, (void*)(data.data()+msg_type_size), sizeof(int), 0);
                            //add msg into the list
                            m_recv_msg.push(data);
                        break;
                        case MSG_FLOAT:
                            data.resize(data.size()+sizeof(float));   
                            ::recv(m_client.m_socket, (void*)(data.data()+msg_type_size), sizeof(float), 0);
                            //add msg into the list
                            m_recv_msg.push(data);
                        break;
                        case MSG_DOUBLE:
                            data.resize(data.size()+sizeof(double));   
                            ::recv(m_client.m_socket, (void*)(data.data()+msg_type_size), sizeof(double), 0);
                            //add msg into the list
                            m_recv_msg.push(data);
                        break;
                        case MSG_STRING:
                        {
                            const size_t size_len   = sizeof(unsigned int);
                            //get len 
                            data.resize(msg_type_size+size_len);  
                            ::recv(m_client.m_socket, (void*)(data.data()+msg_type_size), size_len, 0);
                            int str_len = *((int*)(data.data()+offset_len));
                            //read str 
                            data.resize(msg_type_size+size_len+str_len);  
                            ::recv(m_client.m_socket, (void*)(data.data()+msg_type_size+size_len), str_len, 0);
                            //add msg into the list
                            m_recv_msg.push(data);
                        }
                        break;
                        case MSG_CLOSE_CONNECTION: 
                            //close
                            ::shutdown(m_client.m_socket, SHUT_RDWR);
                            ::close(m_client.m_socket);
                            m_client = socket_info();                              //add msg into the list
                            //add msg into the list
                            m_recv_msg.push(data);
                        break;
                        default: break;
                    }
                }
            }
        }
        
        //close all connection
        void close_server_socket()
        {
            if(m_client.m_socket >= 0)
            {
                //send message
                send_close_connection_immediately();
                #if 1
                //close
                ::shutdown(m_client.m_socket, SHUT_RDWR);
                ::close(m_client.m_socket);
                #endif
            }
            if(m_server.m_socket >= 0)
            {
                set_blocking(m_server.m_socket);
                //disable 
                ::shutdown(m_server.m_socket, SHUT_RDWR);
                //force to close
                ::close(m_server.m_socket);
            }
            //clean
            m_server = socket_info();
            m_client = socket_info();
        }

        //soket info
        socket_info m_client;
        socket_info m_server;
        //send messam_clientge list
        mutable atomic_vector < message_raw > m_send_msg;
        //recv message list
        mutable atomic_vector < message_raw > m_recv_msg;
        //thread info
        std::thread              m_thread;
        std::atomic< bool >      m_run  { 0 };
        std::atomic< int >       m_error{ 0 };
        int                      m_port { 0 };
        
    };
    
}
}
