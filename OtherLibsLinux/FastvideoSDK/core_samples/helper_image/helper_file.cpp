#include <cstdio> 
#include <helper_common.h>

long FileSize(const char *filename) {
	FILE *file;
	FOPEN(file, filename, "rb");
	if (file == NULL)
		return -1;
	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	fclose(file);
	return size;
}
