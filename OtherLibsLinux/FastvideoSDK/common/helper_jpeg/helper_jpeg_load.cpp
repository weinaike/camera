#include <climits>
#include <cstdint>
#include <fstream>
#include <istream>
#include <streambuf>
#include <vector>
#include <cstring>

#include <map>
#include <list>
#include <string>
#include "helper_jpeg.hpp"

#include "FastAllocator.h"
#include "timing.hpp"


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
template<typename T> void Clear(T *data, size_t count){
    memset(data, 0, count * sizeof(T));
}

template<typename T> void Clear(T &data, size_t count){
    memset(&data, 0, count * sizeof(T));
}

template<typename T, unsigned Count> void Clear(T (&data)[Count]){
    Clear(data, Count);
}

template<typename T> void Read(T *dst, size_t count, std::istream &fd){
    for(size_t i = 0; i < count; i++){
        char buf[sizeof(T)];
        fd.read(buf, sizeof(T));

        for(unsigned j = 0; j < sizeof(T); j++)
            reinterpret_cast<char *>(dst + i)[j] = buf[sizeof(T) - 1 - j];
    }
}

template<typename T> void Read(T &dst, std::istream &fd){
    Read(&dst, 1, fd);
}

template<typename T, size_t Count> void Read(T (&dst)[Count], std::istream &fd){
    Read(&dst[0], Count, fd);
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
void ReadDQT(
	fastJpegQuantState_t *state,
	bool(&isUsed)[4],
	std::istream &fd
) {
	uint16_t size;
	uint8_t LqTq;

	Read(size, fd);
	size -= 2;

	for (int j = 0; j < size / 65; j++) {
		Read(LqTq, fd);

		//if ((LqTq >> 4) != 0)
		//	// if = 1, Precision 16 bit
		//	throw FAST_UNSUPPORTED_FORMAT;

		// Precision 8 bit
		unsigned id = (LqTq & 15);

		if (id >= 4)
			throw FAST_INVALID_FORMAT;

		if ((LqTq >> 4) != 0) {
			Read(state->table[id].data, fd);
		} else {
			//Read(state->table[id].data, fd);
			for (int idx = 0; idx < 64; idx++) {
				unsigned char tmp;
				Read(tmp, fd);
				state->table[id].data[idx] = tmp;
			}
		}
		isUsed[id] = true;
	}
}

void ReadDHT(
	fastJpegHuffmanState_t *state,
	bool(&isUsed)[2][2],
	std::istream &fd
) {
	uint16_t size;
	uint8_t TcTh;

	Read(size, fd);
	size -= 2;

	while (size > 0) {
		Read(TcTh, fd);
		uint8_t Tc = (TcTh >> 4) & 15;
		uint8_t Th = (TcTh >> 0) & 15;

		if (Tc >= 2 || Th >= 2)
			throw FAST_INVALID_FORMAT;

		Read(state->table[Tc][Th].bucket, fd);

		uint16_t alphabetSize = 0;
		for (unsigned i = 0; i < MAX_CODE_LEN; i++)
			alphabetSize += state->table[Tc][Th].bucket[i];

		size -= alphabetSize + 17;

		memset(state->table[Tc][Th].alphabet, 0, sizeof(state->table[Tc][Th].alphabet));
		Read(state->table[Tc][Th].alphabet, alphabetSize, fd);

		isUsed[Tc][Th] = true;
	}
}

void ReadSOF0(
	unsigned &H,
	unsigned &W,
	unsigned &BitsPerChannel,
	std::map<uint8_t, unsigned> &labelToChannelPos,
	fastJpegFormat_t &samplingFmt,
	fastJpegScanStruct_t &scanStruct,
	bool isSOF3,
	std::istream &fd
) {
	uint16_t size;
	uint8_t P;
	uint8_t Nf;
	uint16_t H16;
	uint16_t W16;

	Read(size, fd);
	Read(P, fd);
	Read(H16, fd);
	Read(W16, fd);

	Read(Nf, fd);


	H = H16; W = W16;

	if (size != 8 + 3 * Nf)
		throw FAST_INVALID_FORMAT;

	BitsPerChannel = P;
	// if(P != 8)
	//    throw FAST_UNSUPPORTED_FORMAT;

	if (isSOF3)
	{
		if (Nf > 2)
			throw FAST_UNSUPPORTED_FORMAT;
	}
	else if (Nf != 1 && Nf != 3)
		throw FAST_UNSUPPORTED_FORMAT;


	uint8_t C[3], HV[3], Tq[3];

	scanStruct.quantTableMask = 0;

	for (uint8_t channelPos = 0; channelPos < Nf; channelPos++) {
		Read(C[channelPos], fd);
		Read(HV[channelPos], fd);
		Read(Tq[channelPos], fd);
	}

	for (uint8_t channelPos = 0; channelPos < Nf; channelPos++) {
		if (Tq[channelPos] >= 4)
			throw FAST_INVALID_FORMAT;

		scanStruct.quantTableMask |= Tq[channelPos] << (channelPos * BYTE_SIZE);
		labelToChannelPos[C[channelPos]] = channelPos;
	}

	if (Nf == 1 || isSOF3) {
		samplingFmt = FAST_JPEG_Y;

		if (HV[0] != 0x11)
			throw FAST_UNSUPPORTED_FORMAT;
	} else if (Nf == 3) {
		if (HV[0] == 0x11)
			samplingFmt = FAST_JPEG_444;
		else if (HV[0] == 0x21)
			samplingFmt = FAST_JPEG_422;
		else if (HV[0] == 0x22)
			samplingFmt = FAST_JPEG_420;
		else
			throw FAST_UNSUPPORTED_FORMAT;

		if (HV[1] != 0x11 || HV[2] != 0x11)
			throw FAST_UNSUPPORTED_FORMAT;
	} else {
		throw FAST_UNSUPPORTED_FORMAT;
	}

	if (labelToChannelPos.size() != Nf)
		throw FAST_INVALID_FORMAT;
}


void ReadDRI(
	unsigned &restartInterval,
	std::istream &fd
) {
	uint16_t size;
	uint16_t restartInterval16;

	Read(size, fd);
	Read(restartInterval16, fd);
	restartInterval = restartInterval16;

	if (size != 4)
		throw FAST_INVALID_FORMAT;
}

void ReadSOS(
	fastJpegMode_t jpegMode,
	int &predictorClass,
	fastJpegScanStruct_t &scanStruct,
	const std::map<uint8_t, unsigned> &labelToChannelPos,
	std::istream &fd
) {
	uint16_t size;
	uint8_t Ns;

	Read(size, fd);
	Read(Ns, fd);

	if (size != 6 + 2 * Ns)
		throw FAST_INVALID_FORMAT;

	uint8_t C[3], TdTa[3];
	scanStruct.huffmanTableMask[0] = 0;
	scanStruct.huffmanTableMask[1] = 0;
	scanStruct.scanChannelMask = 0;

	std::map<uint8_t, unsigned> labelToScanPos;

	for (uint8_t scanPos = 0; scanPos < Ns; scanPos++) {
		Read(C[scanPos], fd);
		Read(TdTa[scanPos], fd);
		labelToScanPos[C[scanPos]] = scanPos;
	}

	if (labelToChannelPos.size() != labelToScanPos.size())
		throw FAST_UNSUPPORTED_FORMAT;

	std::map<uint8_t, unsigned>::const_iterator
		iChannels = labelToChannelPos.begin(),
		iScan = labelToScanPos.begin();

	for (uint8_t scanPos = 0; scanPos < Ns; scanPos++) {
		if (iChannels++->first != iScan++->first)
			throw FAST_INVALID_FORMAT;

		if (labelToChannelPos.find(C[scanPos]) == labelToChannelPos.end())
			throw FAST_INVALID_FORMAT;

		scanStruct.huffmanTableMask[0] |= ((TdTa[scanPos] >> 4) & 15) << (scanPos * BYTE_SIZE);
		scanStruct.huffmanTableMask[1] |= ((TdTa[scanPos] >> 0) & 15) << (scanPos * BYTE_SIZE);
		scanStruct.scanChannelMask |= labelToChannelPos.find(C[scanPos])->second << (scanPos * BYTE_SIZE);
	}

	scanStruct.scanGroupMask = Ns;

	{
		uint8_t Ss, Se, A;
		Read(Ss, fd);
		Read(Se, fd);
		Read(A, fd);

		if (jpegMode == FAST_JPEG_LOSSLESS) {
			if (Ss > 7 || Ss == 0 || Se != 0 || A > 15)
				throw FAST_UNSUPPORTED_FORMAT;
			predictorClass = Ss;
		} else if (jpegMode == FAST_JPEG_SEQUENTIAL_DCT) {
			if (Ss != 0 || Se != 63 || A != 0)
				throw FAST_UNSUPPORTED_FORMAT;
			predictorClass = -1;
		} else
			throw FAST_UNSUPPORTED_FORMAT;
	}
}

unsigned FileRemainingLength(std::istream &fd) {
	std::streampos curr = fd.tellg();
	fd.seekg(0, fd.end);
	std::streampos end = fd.tellg();
	fd.seekg(curr);

	if ((end - curr) > UINT_MAX)
		throw FAST_UNSUPPORTED_FORMAT;

	return unsigned(end - curr);
}

fastStatus_t jfifLoadHeader(
	fastJfifInfo_t *jfifInfo,
	std::istream &fd
) {
	jfifInfo->restartInterval = 0;
	bool isQuantUsed[4], isHuffmanUsed[2][2];
	bool isSOF0 = false;

	Clear(jfifInfo->quantState, 1);
	Clear(jfifInfo->huffmanState, 1);
	Clear(isQuantUsed);
	Clear(isHuffmanUsed);

	std::vector< fastJpegExifSection_t> exifData;

	bool restartIntervalFound = false;

	try {
		//Find SOI
		{
			bool isPrevFF = false;

			while (1) {
				uint8_t curr;
				fd.read((char *)&curr, 1);

				if (isPrevFF && (curr == 0xD8U))
					break;

				isPrevFF = (curr == 0xFFU);
			}
		}

		{
			bool isPrevFF = false;
			unsigned channelCount = 0;
			std::map<uint8_t, unsigned> channelMap;

			while (!fd.eof()) {
				uint8_t curr;
				fd.read((char *)&curr, 1);

				if (isPrevFF && (curr != 0xFF))
					switch (curr) {
						case 0xe0:
						case 0xe1:	// used for EXIF data
						case 0xe2:
						case 0xe3:
						case 0xe4:
						case 0xe5:
						case 0xe6:
						case 0xe7:
						case 0xe8:
						case 0xe9:
						case 0xea:
						case 0xeb:
						case 0xec:
						case 0xed:
						case 0xee:
						case 0xef:
						case 0xfe:
						{
							uint16_t size;
							fastJpegExifSection_t exif = { 0 };

							Read(size, fd);
							size -= 2;
							//fd.seekg(size, fd.cur);
							exif.exifCode = uint16_t(0xFF00 + curr);
							exif.exifData = (char *)malloc((size + 1) * sizeof(char));
							Read(exif.exifData, size, fd);
							exif.exifData[size] = '\0';
							exif.exifLength = size;

#ifdef _DEBUG_LOG
							printf("Exif data: code = 0x%.4X, size = %d, data = %s\n", exif.exifCode, exif.exifLength, exif.exifData);
#endif

							exifData.push_back(exif);
							break;
						}
						case 0xDBU:
							ReadDQT(&jfifInfo->quantState, isQuantUsed, fd);
							break;

						case 0xC4:
							ReadDHT(&jfifInfo->huffmanState, isHuffmanUsed, fd);
							break;

						case 0xDD:
							ReadDRI(jfifInfo->restartInterval, fd);
							restartIntervalFound = true;
							break;

						case 0xC0:
						case 0xC1:
						case 0xC3:

							if (curr == 0xC3)
								jfifInfo->jpegMode = FAST_JPEG_LOSSLESS;
							else
								jfifInfo->jpegMode = FAST_JPEG_SEQUENTIAL_DCT;


							ReadSOF0(jfifInfo->height, jfifInfo->width, jfifInfo->bitsPerChannel, channelMap, jfifInfo->jpegFmt, jfifInfo->scanMap, curr == 0xC3, fd);
							isSOF0 = true;
							break;

						case 0xC5:
#ifdef CONSOLE_APP
							fprintf(stderr, "JPEG library does not support sequential DCT\n");
#endif
							return FAST_UNSUPPORTED_FORMAT;

						case 0xC2:
						case 0xC6:
#ifdef CONSOLE_APP
							fprintf(stderr, "JPEG library does not support progressive DCT\n");
#endif
							return FAST_UNSUPPORTED_FORMAT;


						case 0xC7:
#ifdef CONSOLE_APP
							fprintf(stderr, "JPEG library does not support lossless format\n");
#endif
							return FAST_UNSUPPORTED_FORMAT;

						case 0xC8:
						case 0xC9:
						case 0xCA:
						case 0xCB:
						case 0xCD:
						case 0xCE:
						case 0xCF:
#ifdef CONSOLE_APP
							fprintf(stderr, "JPEG library does not arithmetic coding\n");
#endif
							return FAST_UNSUPPORTED_FORMAT;


						case 0xDA:
						{
							if (!isSOF0) {
#ifdef CONSOLE_APP
								fprintf(stderr, "Scan marker was not found in file\n");
#endif
								return FAST_INVALID_FORMAT;
							}

							if (!restartIntervalFound) {
#ifdef _DEBUG_LOG
								fprintf(stderr, "Restart Interval was not set in file\n");
#endif
								jfifInfo->restartInterval = 0;
							}

							ReadSOS(jfifInfo->jpegMode, jfifInfo->predictorClass, jfifInfo->scanMap, channelMap, fd);

							//Check if all referenced Huffman tables of a scan are present
							for (uint8_t scanPos = 0; scanPos < channelCount; scanPos++)
								for (uint8_t Tc = 0; Tc < 2; Tc++)
									if (!isHuffmanUsed[Tc][Map_host(jfifInfo->scanMap.huffmanTableMask[Tc], scanPos)]) {
#ifdef CONSOLE_APP
										fprintf(stderr, "Huffman tables verifying error\n");
#endif
										return FAST_INVALID_FORMAT;
									}

							//Check if all referenced quantization tables of an image are present
							for (uint8_t channelPos = 0; channelPos < channelCount; channelPos++)
								if (!isQuantUsed[Map_host(jfifInfo->scanMap.quantTableMask, channelPos)]) {
#ifdef CONSOLE_APP
									fprintf(stderr, "Quantization tables verifying error\n");
#endif
									return FAST_INVALID_FORMAT;
								}

							//
							jfifInfo->exifSectionsCount = static_cast<unsigned int>(exifData.size());
							jfifInfo->exifSections = (fastJpegExifSection_t *)malloc(exifData.size() * sizeof(fastJpegExifSection_t));

							int idx = 0;
							for (std::vector<fastJpegExifSection_t>::iterator i = exifData.begin(); i != exifData.end(); i++) {
								jfifInfo->exifSections[idx++] = (*i);
							}

							jfifInfo->headerSize = static_cast<unsigned int>(fd.tellg());
							return FAST_OK;
						}
						break;

						case 0xD9:
							goto brk;
							break;

						default:
							if ((curr != 0) && (curr < 0xD0) && (curr > 0xD7)) {
#ifdef _DEBUG_LOG
								printf("Skipped marker %x\n", curr);
#endif
							}
				}

				isPrevFF = (curr == 0xFFU);
			}

		brk:;
		}

	}
	catch (...) {
#ifdef _DEBUG
		fprintf(stderr, "Input file has inappropriate format\n");
#endif
		return FAST_INVALID_FORMAT;
	}

	jfifInfo->exifSectionsCount = static_cast<unsigned int>(exifData.size());
	jfifInfo->exifSections = (fastJpegExifSection_t *)malloc(exifData.size() * sizeof(fastJpegExifSection_t));

	int idx = 0;
	for (std::vector<fastJpegExifSection_t>::iterator i = exifData.begin(); i != exifData.end(); i++) {
		jfifInfo->exifSections[idx++] = (*i);
	}

	return FAST_OK;
}

fastStatus_t fastJfifHeaderLoadFromFile(const char *filename, fastJfifInfo_t *jfifInfo) {
	std::ifstream fd(filename, std::ios::in | std::ios::binary);
	if (fd.is_open()) {
		fd.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);

		fastStatus_t res = jfifLoadHeader(jfifInfo, fd);

		unsigned streamSize = FileRemainingLength(fd);
		jfifInfo->bytestreamSize = streamSize;

		return res;
	}

	return FAST_IO_ERROR;
}

fastStatus_t fastJfifBytestreamLoadFromFile(const char *filename, fastJfifInfo_t *jfifInfo) {
	std::ifstream fd(filename, std::ios::in | std::ios::binary);
	if (fd.is_open()) {
		fd.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);

		fd.seekg(0, fd.end);
		std::streampos end = fd.tellg();

		if (jfifInfo->headerSize >= end)
			return FAST_INVALID_SIZE;

		fd.seekg(jfifInfo->headerSize, fd.beg);

		unsigned streamSize = (unsigned)end - jfifInfo->headerSize;

		if (jfifInfo->bytestreamSize < streamSize) {
			return FAST_INVALID_SIZE;
		}

		jfifInfo->bytestreamSize = streamSize;

		fd.read((char *)(jfifInfo->h_Bytestream), jfifInfo->bytestreamSize);

		return jfifLoadHeader(jfifInfo, fd);
	}

	return FAST_IO_ERROR;
}

struct membuf :std::streambuf {
	membuf(char* b, char* e) { this->setg(b, b, e); }

	virtual std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way,
								   std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) {
		if ((which & std::ios_base::in) == 0) {
			// unused type
			return -1;
		}

		std::streamoff baseOffset = 0;
		if (way == std::ios_base::beg) {
			// bytes to start
			baseOffset = this->eback() - this->gptr();
		} else if (way == std::ios_base::end) {
			// bytes to end
			baseOffset = this->in_avail();//this->egptr() - this->gptr();
		};

		if ((baseOffset + off) > 0) {
			this->gbump(static_cast<int>(baseOffset + off));
		}

		return (this->gptr() - this->eback());
	};
};

fastStatus_t fastJfifHeaderLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
) {
	membuf sbuf((char *)inputStream, (char *)(inputStream + inputStreamSize));
	std::istream fd(&sbuf);
	fd.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);

	fastStatus_t res = jfifLoadHeader(jfifInfo, fd);

	unsigned streamSize = inputStreamSize - jfifInfo->headerSize;
	jfifInfo->bytestreamSize = streamSize;
	return res;
}


fastStatus_t fastJfifLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
) {
	membuf sbuf((char *)inputStream, (char *)(inputStream + inputStreamSize));
	std::istream fd(&sbuf);
	fd.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);

	fastStatus_t res = jfifLoadHeader(jfifInfo, fd);
	if (res != FAST_OK)
		return res;

	if (jfifInfo->h_Bytestream == 0)
		return FAST_INVALID_VALUE;

	if (jfifInfo->headerSize >= inputStreamSize)
		return FAST_INVALID_SIZE;

	unsigned streamSize = inputStreamSize - jfifInfo->headerSize;

	if (jfifInfo->bytestreamSize < streamSize) {
		return FAST_INVALID_SIZE;
	}

	jfifInfo->bytestreamSize = streamSize;

	memcpy((char *)(jfifInfo->h_Bytestream), &inputStream[jfifInfo->headerSize], jfifInfo->bytestreamSize);

	return FAST_OK;
}

fastStatus_t fastJfifBytestreamLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
) {
	if (jfifInfo->h_Bytestream == 0)
		return FAST_INVALID_VALUE;

	if (jfifInfo->headerSize >= inputStreamSize)
		return FAST_INVALID_SIZE;

	unsigned streamSize = inputStreamSize - jfifInfo->headerSize;

	if (jfifInfo->bytestreamSize < streamSize) {
		return FAST_INVALID_SIZE;
	}

	jfifInfo->bytestreamSize = streamSize;

	memcpy((char *)(jfifInfo->h_Bytestream), &inputStream[jfifInfo->headerSize], jfifInfo->bytestreamSize);

	return FAST_OK;
}

void fastJfifFreeExif(fastJfifInfo_t* jfifInfo)
{
	for (unsigned i = 0; i < jfifInfo->exifSectionsCount; i++) {
		free(jfifInfo->exifSections[i].exifData);
	}

	if (jfifInfo->exifSections != NULL) {
		free(jfifInfo->exifSections);
	}
}
