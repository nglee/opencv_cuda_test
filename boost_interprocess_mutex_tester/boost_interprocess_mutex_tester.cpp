#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <iostream>

using namespace std;
using namespace boost::interprocess;

void first_process(named_mutex& mtx);
void second_process(named_mutex& mtx);

int main()
{
	try {
		named_mutex mtx(create_only, "test_named_mutex");
		first_process(mtx);
	} catch (interprocess_exception& e) {
        cout << e.what() << endl;
		named_mutex mtx(open_only, "test_named_mutex");
		second_process(mtx);
	}
    //named_mutex::remove("test_named_mutex");
}

void first_process(named_mutex& mtx) throw(interprocess_exception)
{
	named_semaphore sema(open_or_create, "test_named_semaphore", 0);
	mtx.lock();
	sema.post();
}

void second_process(named_mutex& mtx) throw(interprocess_exception)
{
	named_semaphore sema(open_or_create, "test_named_semaphore", 0);
	sema.wait();
	mtx.lock();
}