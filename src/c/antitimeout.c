#include "parse.h"
#include <stdio.h>
#include <stdlib.h>

char path[] = "D:\\CDR\\progress.json";
char buffer[100];

progress *state;
void readJson();

int main(int argc, char const *argv[])
{
    do
    {
        
    } while (1);
    
    readJson();
    system("pause");
    return 0;
}

void readJson()
{
    FILE *file = fopen(path, "r");
    int i = 0;
    while ((fread(&buffer[i], sizeof(char), 1, file)) > 0)
        i++;

    puts(buffer);
    state = parse(buffer);
    fclose(file);

    printf("folder=%d file=%d", getFile(state), getFolder(state));
}
