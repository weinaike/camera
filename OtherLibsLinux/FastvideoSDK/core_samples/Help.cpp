#include <cstdio>

#ifndef __GNUC__
static const char *helpCompiledFor = "Compiled for Windows-10 [x64]\n";
#else
static const char *helpCompiledFor = "Compiled for Linux\n";
#endif
static const char *helpCommonInfo =
	"\n" \
	"This software is prepared for non-commercial use only. It is free for personal and educational (including non-profit organization) use. Distribution of this software without any permission from IP owner is NOT allowed. NO warranty and responsibility is provided by the authors for the consequences of using it.\n" \
	"\n" \
	"Hardware Requirements\n" \
	" - NVIDIA GeForce / Quadro / Tesla GPU with Compute Capability >= 5.0, NVIDIA drivers 528.02 or later. " \
	"For the latest NVIDIA drivers visit https://www.nvidia.com/Download/index.aspx\n" \
	"\n";
static const char *sdkName = " for GPU Image and Video Processing SDK\n";

extern const char *projectName;
extern const char *helpProject;

void helpPrint(void) {
	printf("%s", projectName);
	printf("%s", sdkName);
	printf("%s", helpCompiledFor);
	printf("%s", helpCommonInfo);
	printf("%s", helpProject);
}
