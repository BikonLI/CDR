#include "parse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

progress *parse(const char *string)
{
    int second_start_index;
    progress *result = (progress *)malloc(sizeof(progress));
    sscanf(string, "{\"folder\": %d, \"file\": %d}", 
    &result->folder, &result->file);

    return result;
}

int getFile(const progress *json)
{
    return json->file;
}

int getFolder(const progress *json)
{
    return json->folder;
}