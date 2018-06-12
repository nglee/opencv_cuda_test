NAME    = boost_interprocess_delete_destroy_tester
SRC     = $(NAME).cpp
OUT     = $(NAME)
CXX     = g++

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) -o $@ -std=c++14 -I/usr/local/boost-1.60.0/include -L/usr/local/boost-1.60.0/lib -pthread $< -g
	$(CXX) -o deleter -std=c++14 -I/usr/local/boost-1.60.0/include -L/usr/local/boost-1.60.0/lib -pthread deleter.cpp

clean:
	rm $(OUT)
