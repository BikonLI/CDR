#ifndef PARSE_H
#define PARSE_H

typedef struct
{
    int file;
    int folder;
} progress;

progress *parse(const char *string);
int getFile(const progress *json);
int getFolder(const progress *json);

#endif
