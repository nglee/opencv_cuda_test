#include <boost/interprocess/xsi_key.hpp>
#include <boost/interprocess/managed_xsi_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

#include <iostream>

using namespace boost::interprocess;

void first_process(managed_xsi_shared_memory *shm)
{
    pid_t pid = ::getpid();

    struct shm_remove
    {
        pid_t pid;
        shm_remove(pid_t pid_) : pid(pid_) {}
        ~shm_remove() {
            try {
                managed_xsi_shared_memory shm{ open_only, xsi_key("Makefile", 239) };
                managed_xsi_shared_memory::remove(shm.get_shmid());
            } catch (interprocess_exception& e) {
                std::cout << std::to_string(pid) << ": (" << e.what() << ")" << std::endl;
            }
        }
    } remover(pid);

    interprocess_mutex *mtx = shm->find_or_construct<interprocess_mutex>("gMutex")();
    interprocess_semaphore *sema1 = shm->find_or_construct<interprocess_semaphore>("gSema1")(0);
    interprocess_semaphore *sema2 = shm->find_or_construct<interprocess_semaphore>("gSema2")(0);

    std::cout << std::to_string(pid) << ": gMutex, gSema1&2 created, will wait for gSema1 post" << std::endl;
    sema1->wait();
    std::cout << std::to_string(pid) << ": gSema1 post recieved, will delete gMutex" << std::endl;
    delete mtx;
    std::cout << std::to_string(pid) << ": gMutex deleted, will post gSema2" << std::endl;
    sema2->post();
    std::cout << std::to_string(pid) << ": gSema2 posted, will delete sema1&2" << std::endl;

    delete sema1;
    delete sema2;
}

void second_process(managed_xsi_shared_memory *shm)
{
    pid_t pid = ::getpid();

    struct shm_remove
    {
        pid_t pid;
        shm_remove(pid_t pid_) : pid(pid_) {}
        ~shm_remove() {
            try {
                managed_xsi_shared_memory shm{ open_only, xsi_key("Makefile", 239) };
                managed_xsi_shared_memory::remove(shm.get_shmid());
            } catch (interprocess_exception& e) {
                std::cout << std::to_string(pid) << ": (" << e.what() << ")" << std::endl;
            }
        }
    } remover(pid);

    try {
        shm = new managed_xsi_shared_memory{ open_only, xsi_key("Makefile", 239) };
    } catch (interprocess_exception& e) {
        std::cout << std::to_string(pid) << ": shared memory open fail(" << e.what() << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    interprocess_mutex *mtx = shm->find_or_construct<interprocess_mutex>("gMutex")();
    interprocess_semaphore *sema1 = shm->find_or_construct<interprocess_semaphore>("gSema1")(0);
    interprocess_semaphore *sema2 = shm->find_or_construct<interprocess_semaphore>("gSema2")(0);

    std::cout << std::to_string(pid) << ": gMutex, gSema1&2 created, will lock gMutex" << std::endl;
    mtx->lock();
    std::cout << std::to_string(pid) << ": gMutex locked, will post gSema1" << std::endl;
    sema1->post();
    std::cout << std::to_string(pid) << ": gSema1 posted, will wait for gSema2 post" << std::endl;
    sema2->wait();
    std::cout << std::to_string(pid) << ": gSema2 post recieved, will unlock gMutex" << std::endl;
    mtx->unlock();
    std::cout << std::to_string(pid) << ": gMutex unlocked, will delete gMutex, gSema1&2" << std::endl;

    delete mtx;
    delete sema1;
    delete sema2;
}

int main()
{
    pid_t pid = ::getpid();
    managed_xsi_shared_memory *shm;

    try {
        shm = new managed_xsi_shared_memory{ create_only, xsi_key("Makefile", 239), 1024 };
        std::cout << std::to_string(pid) << ": shared memory created" << std::endl;
        first_process(shm);
    } catch (interprocess_exception& e) {
        std::cout << std::to_string(pid) << ": shared memory create fail(" << e.what() << ")"
            << ", will open it instead" << std::endl;
        second_process(shm);
    }

    return EXIT_SUCCESS;
}

