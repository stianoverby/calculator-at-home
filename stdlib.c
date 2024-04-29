#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
/*
    Source code used to implement the stdlib for our language.
    https://godbolt.org/ was used for translating the code to 
    x86_64 assembler.
*/
void print_number(int64_t x){
    char buf[32];
    size_t buf_size = 1;
    int is_negative = 0;
    if (x < 0) {
        is_negative = 1;
        x = -x;
    }
    buf[sizeof(buf) - buf_size] = '\n';
    do {
        buf[sizeof(buf) - buf_size - 1] = x % 10 + '0';
        buf_size++;
        x /= 10;
    } while(x);

    if (is_negative) {
            buf[sizeof(buf) - buf_size - 1] = '-';
            buf_size++;
        }

    write(1, &buf[sizeof(buf) - buf_size], buf_size);
}