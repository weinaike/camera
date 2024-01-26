#ifndef __HELPER_DIR__
#define __HELPER_DIR__

#include <list>
#include <string>

bool compareFileExtension(const char *fileName, const char *etalon);
bool fileExist(const char *fileName);
int fileSize(const char *fname);
int getFileList(const char *folderName, std::list<std::string> &fileList);

#endif // __HELPER_DIR__
