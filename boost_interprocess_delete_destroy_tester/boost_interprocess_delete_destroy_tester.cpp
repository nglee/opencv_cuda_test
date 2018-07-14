
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

#ifdef _WIN32

#include <boost/interprocess/managed_windows_shared_memory.hpp>
typedef boost::interprocess::managed_windows_shared_memory smem_t;
typedef int pid_t;
#define getpid _getpid

#else

#include <boost/interprocess/xsi_key.hpp>
#include <boost/interprocess/managed_xsi_shared_memory.hpp>
typedef boost::interprocess::managed_xsi_shared_memory smem_t

#endif

#include <iostream>

using namespace boost::interprocess;

void first_process(smem_t *shm)
{
    pid_t pid = ::getpid();

    //struct shm_remove
    //{
    //    pid_t pid;
    //    shm_remove(pid_t pid_) : pid(pid_) {}
    //    ~shm_remove() {
    //        try {
    //            smem_t shm{ open_only, xsi_key("Makefile", 239) };
    //            smem_t::remove(shm.get_shmid());
    //        } catch (interprocess_exception& e) {
    //            std::cout << std::to_string(pid) << ": (" << e.what() << ")" << std::endl;
    //        }
    //    }
    //} remover(pid);

    interprocess_mutex *mtx = shm->find_or_construct<interprocess_mutex>("gMutex")();
    interprocess_semaphore *sema1 = shm->find_or_construct<interprocess_semaphore>("gSema1")(0);
    interprocess_semaphore *sema2 = shm->find_or_construct<interprocess_semaphore>("gSema2")(0);

    std::cout << std::to_string(pid) << ": gMutex, gSema1&2 created, will wait for gSema1 post" << std::endl;
    sema1->wait();
    std::cout << std::to_string(pid) << ": gSema1 post recieved, will delete gMutex" << std::endl;
    delete mtx; // crashes, if shm->destroy_ptr(mtx), does not crash (https://stackoverflow.com/a/51335826/7724939)
    std::cout << std::to_string(pid) << ": gMutex deleted, will post gSema2" << std::endl;
    sema2->post();
    std::cout << std::to_string(pid) << ": gSema2 posted, will delete sema1&2" << std::endl;

    //delete sema1;
    //delete sema2;
}

void second_process(smem_t *shm)
{
    pid_t pid = ::getpid();

    //struct shm_remove
    //{
    //    pid_t pid;
    //    shm_remove(pid_t pid_) : pid(pid_) {}
    //    ~shm_remove() {
    //        try {
    //            smem_t shm{ open_only, xsi_key("Makefile", 239) };
    //            smem_t::remove(shm.get_shmid());
    //        } catch (interprocess_exception& e) {
    //            std::cout << std::to_string(pid) << ": (" << e.what() << ")" << std::endl;
    //        }
    //    }
    //} remover(pid);

    try {
#ifdef _WIN32
        shm = new smem_t{ open_only, "boost_interprocess_delete_destroy_tester" };
#else
        shm = new smem_t{ open_only, xsi_key("Makefile", 239) };
#endif
    } catch (std::exception& e) {
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

    //delete mtx;
    //delete sema1;
    //delete sema2;
}

int main()
{
    pid_t pid = ::getpid();
    smem_t *shm = nullptr;

    try {
#ifdef _WIN32
        shm = new smem_t{ create_only, "boost_interprocess_delete_destroy_tester", 1024 };
#else
        shm = new smem_t{ create_only, xsi_key("Makefile", 239), 1024 };
#endif
        std::cout << std::to_string(pid) << ": shared memory created" << std::endl;
        first_process(shm);
    } catch (interprocess_exception& e) {
        std::cout << std::to_string(pid) << ": shared memory create fail(" << e.what() << ")"
            << ", will open it instead" << std::endl;
        second_process(shm);
    }

    return EXIT_SUCCESS;
}

