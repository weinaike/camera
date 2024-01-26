#ifndef __DUMP_IMAGE__
#define __DUMP_IMAGE__

template<typename T>
int DumpImageChannelToText(
	T *img,
	const unsigned pitch,

	const unsigned width,
	const unsigned height,

	const char *fnameR,
	const char *fnameG,
	const char *fnameB
);

#endif // __DUMP_IMAGE__
