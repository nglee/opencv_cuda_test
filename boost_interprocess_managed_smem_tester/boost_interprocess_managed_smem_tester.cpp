#include <boost/interprocess/managed_windows_shared_memory.hpp>
#include <iostream>

using namespace boost::interprocess;

int main()
{
    managed_windows_shared_memory *shm = nullptr;

    try {
        shm = new managed_windows_shared_memory{ create_only, "shm", 1024 };
        std::cout << "shared memory created" << std::endl;
    } catch (interprocess_exception& e) {
        std::cout << e.what() << std::endl;
        shm = new managed_windows_shared_memory{ open_only, "shm" };
        std::cout << "shared memory creation failed, opened instead" << std::endl;
    }
}