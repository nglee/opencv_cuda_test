#include <boost/interprocess/xsi_key.hpp>
#include <boost/interprocess/managed_xsi_shared_memory.hpp>

#include <iostream>
#include <string>

using namespace boost::interprocess;

int main()
{
    managed_xsi_shared_memory *shm;
    int pid = ::getpid();

    try {
        shm = new managed_xsi_shared_memory{ open_only, xsi_key("Makefile", 239) };
    } catch (interprocess_exception& e) {
        std::cout << std::to_string(pid) << ": " << "shared memory open fail (" << e.what() << ")\n";
        exit(EXIT_FAILURE);
    }

    shm->remove(shm->get_shmid());

    return EXIT_SUCCESS;
}
