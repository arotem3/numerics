CXX = g++
CXXFLAGS = -g -Wall -fPIC
LDFLAGS = -shared
LIBS = -larmadillo
DEPS = numerics.hpp statistics.hpp
TARGET = statistics.so

SRCS = chi_sq.cpp distributions.cpp functions.cpp tests.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: ${TARGET}

$(TARGET): $(OBJS)
	$(CXX) ${LDFLAGS} -o $@ $^

$(SRCS:.cpp=.d):%.d:%.cpp
	$(CXX) $(CXXFLAGS) -MM $< >$@

include $(SRCS:.cpp=.d)

.PHONY: clean
clean:
	rm ${TARGET} ${OBJS} $(SRCS:.cpp=.d)