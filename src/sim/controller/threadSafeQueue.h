#pragma once
#include <queue>
#include <mutex>
#include <thread>

namespace waterSim::sim{
    template<typename T>
    class threadSafeQueue{
    private:
        std::queue<T> queue;
        std::mutex mutex;
    public:
        void push(T& t){
            mutex.lock();
            queue.push(t);
            mutex.unlock();
        }
        void push(T&& t){
            mutex.lock();
            queue.push(std::move(t));
            mutex.unlock();
        }
        T pop(){
            while(true){
                mutex.lock();
                if(queue.empty()){
                    mutex.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                } else {
                    break;
                }
            }
            T front = queue.front();
            queue.pop();
            mutex.unlock();
            return front;
        }
    };
}