
#include <cstdio>

#include "helper_quant_table.hpp"
#include "helper_common.h"
#include "helper_jpeg/helper_jpeg.hpp"

fastStatus_t fvLoadQuantTable(const char *file, fastJpegQuantState_t& quantState) {
	FILE *fp = NULL;

	quantState = { 0 };
	if (FOPEN_FAIL(FOPEN(fp, file, "r")))
		return FAST_IO_ERROR;

	for (int i = 0; i < 2; i++) {
		unsigned j = 0;
		while (j < DCT_SIZE * DCT_SIZE && !feof(fp)) {
			int value;
			fscanf(fp, "%d", &value);
			quantState.table[i].data[j] = value;
			j++;
		}
	}
	fclose(fp);

	return FAST_OK;
}
