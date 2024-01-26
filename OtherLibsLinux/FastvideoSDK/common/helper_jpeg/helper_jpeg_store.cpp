#include <cassert>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>

#include "fastvideo_sdk_define.h"
#include "helper_jpeg.hpp"
#include "JpegCommonDefines.hpp"

class JfifBytestream {
private:
	std::vector<uint8_t> data_;

public:
	JfifBytestream() {}

	template<typename T> JfifBytestream &operator<<(const T &data) {
		const uint8_t *raw = reinterpret_cast<const uint8_t *>(&data);

		for (unsigned i = 0; i < sizeof(T); i++)
			data_.push_back(raw[sizeof(T) - 1 - i]);

		return *this;
	}

	template<typename T, unsigned N> JfifBytestream &operator<<(const T(&data)[N]) {
		for (unsigned i = 0; i < N; i++)
			(*this) << data[i];

		return *this;
	}

	const uint8_t *GetBase(void) {
		return data_.data();
	}

	size_t GetSize(void) {
		return data_.size();
	}

	~JfifBytestream(void) {
		data_.clear();
	}
};

////////////////////////////////////////////////////////////////////////////////
// Store
////////////////////////////////////////////////////////////////////////////////
void AppendDQT(
	JfifBytestream &stream,

	const fastQuantTable_t &q,
	uint8_t Pq,
	uint8_t Tq
) {
	assert(Tq < 4);
	assert(Pq == 0 || Pq == 1);

	stream << uint16_t(0xFFDB) << uint16_t(2 + 65 + 64 * Pq) << uint8_t((Pq << 4) | Tq);
	if (Pq == 1) {
		stream << q.data;
	} else {
		for (int i = 0; i < DCT_SIZE * DCT_SIZE; i++) {
			uint8_t value = static_cast<uint8_t>(q.data[i]);
			stream << value;
		}
	}
}

void AppendDHT(
	JfifBytestream &stream,

	const fastHuffmanTable_t &table,
	uint8_t Tc,
	uint8_t Th
) {
	uint16_t alphabetSize = 0;
	{
		for (unsigned i = 0; i < MAX_CODE_LEN; i++)
			alphabetSize += table.bucket[i];
	}

	assert(Tc < 2 && Th < 2);

	stream << uint16_t(0xFFC4) << uint16_t(19 + alphabetSize) << uint8_t((Tc << 4) | Th) << table.bucket;

	for (unsigned i = 0; i < alphabetSize; i++)
		stream << table.alphabet[i];
}

uint8_t ChannelLabelFromChannelPos(uint8_t channelPos){
    return channelPos + 1;
}

void AppendSOF0(
	JfifBytestream &stream,

	uint16_t H,
	uint16_t W,
	unsigned bitsPerChannel,

	const fastJpegScanStruct_t &scanStruct,
	fastJpegFormat_t samplingFmt
) {
	const uint8_t Nf = (samplingFmt == FAST_JPEG_Y) ? 1 : 3;

	stream << uint16_t(bitsPerChannel == 8 ? 0xFFC0 : 0xFFC1) << uint16_t(8 + 3 * Nf) << uint8_t(bitsPerChannel) << H << W << Nf;

	const unsigned channelMbHeight[3] = {
		(samplingFmt == FAST_JPEG_420) ? 2u : 1u,
		1u,
		1u
	};

	const unsigned channelMbWidth[3] = {
		(samplingFmt == FAST_JPEG_420 || samplingFmt == FAST_JPEG_422) ? 2u : 1u,
		1u,
		1u
	};

	for (uint8_t channelPos = 0; channelPos < Nf; channelPos++) {
		const uint8_t  Ci = ChannelLabelFromChannelPos(channelPos);
		const uint8_t HVi = static_cast<uint8_t>((channelMbWidth[channelPos] << 4) | channelMbHeight[channelPos]);
		const uint8_t Tqi = static_cast<uint8_t>(Map_host(scanStruct.quantTableMask, channelPos));
		stream << Ci << HVi << Tqi;
	}
}

void AppendSOS(
	JfifBytestream &stream,
	fastJpegFormat_t samplingFmt,
	const fastJpegScanStruct_t &scanStruct
) {
	const uint8_t Ns = (samplingFmt == FAST_JPEG_Y) ? 1 : 3;
	stream << uint16_t(0xFFDA) << uint16_t(6 + 2 * Ns) << Ns;

	for (uint8_t scanPos = 0; scanPos < Ns; scanPos++) {
		const uint8_t    Ci = ChannelLabelFromChannelPos(static_cast<uint8_t>(Map_host(scanStruct.scanChannelMask, scanPos)));
		const uint8_t TdTai = static_cast<uint8_t>((Map_host(scanStruct.huffmanTableMask[0], scanPos) << 4) | Map_host(scanStruct.huffmanTableMask[1], scanPos));
		stream << Ci << TdTai;
	}

	{
		const uint8_t   Ss = 0;
		const uint8_t   Se = 63;
		const uint8_t    A = 0x00;
		stream << Ss << Se << A;
	}
}

void AppendSOI(JfifBytestream &writer) {
	writer << uint16_t(0xFFD8);
}

void AppendDRI(
	JfifBytestream &stream,
	uint16_t    restartInterval
) {
	stream << uint16_t(0xFFDD) << uint16_t(4) << restartInterval;
}

void AppendAPP0(
	JfifBytestream &stream
) {
	const uint8_t data[14] = { 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00 };

	stream << uint16_t(0xFFE0) << uint16_t(16) << data;
}

void AppendAPP(
	JfifBytestream &stream,
	fastJpegExifSection_t &exif
) {
	stream << uint16_t(exif.exifCode) << uint16_t(exif.exifLength + 2);
	for (int i = 0; i < exif.exifLength; i++) {
		stream << exif.exifData[i];
	}
}

void AppendHeader(
	JfifBytestream &stream,

	uint16_t imageH,
	uint16_t imageW,
	unsigned bitsPerChannel,

	const fastJpegQuantState_t   *quantState,
	const fastJpegHuffmanState_t *huffmanState,
	const fastJpegScanStruct_t      *scanMap,

	fastJpegFormat_t samplingFmt,
	uint16_t restartInterval,
	fastJpegExifSection_t *exifSections,
	unsigned exifSectionsCount
) {
	AppendSOI(stream);
	if (exifSections == NULL || exifSectionsCount == 0) {
		AppendAPP0(stream);
	} else {
		for (unsigned int i = 0; i < exifSectionsCount; i++) {
			AppendAPP(stream, exifSections[i]);
		}
	}

	bool isQuant[4], isHuffman[2][2];

	memset(isQuant, 0, sizeof(isQuant));
	memset(isHuffman, 0, sizeof(isHuffman));

	const unsigned ChannelCount = (samplingFmt == FAST_JPEG_Y) ? 1 : 3;

	for (uint8_t channelPos = 0; channelPos < ChannelCount; channelPos++)
		isQuant[Map_host(scanMap->quantTableMask, channelPos)] = true;

	for (uint8_t classID = 0; classID < 2; classID++)
		for (uint8_t scanPos = 0; scanPos < ChannelCount; scanPos++)
			isHuffman[classID][Map_host(scanMap->huffmanTableMask[classID], scanPos)] = true;

	for (uint8_t quantID = 0; quantID < 4; quantID++)
		if (isQuant[quantID])
			AppendDQT(stream, quantState->table[quantID], bitsPerChannel == 12 ? 1 : 0, quantID);

	AppendSOF0(stream, imageH, imageW, bitsPerChannel, *scanMap, samplingFmt);

	for (uint8_t classID = 0; classID < 2; classID++)
		for (uint8_t entropyID = 0; entropyID < 2; entropyID++)
			if (isHuffman[classID][entropyID])
				AppendDHT(stream, huffmanState->table[classID][entropyID], classID, entropyID);

	AppendDRI(stream, restartInterval);
	AppendSOS(stream, samplingFmt, *scanMap);
}

template<typename T> void Write(std::ofstream &fd, const T *data, size_t count = 1) {
	if (count > 0)
		fd.write(reinterpret_cast<const char *>(data), count * sizeof(T));
}

fastStatus_t fastJfifStoreToFile(
    const char *filename,
    fastJfifInfo_t *jfifInfo
){
    try {
        std::ofstream fd(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        fd.exceptions(std::ios::failbit | std::ios::badbit);

		JfifBytestream header;
        AppendHeader(
            header,

			static_cast<uint16_t>(jfifInfo->height),
			static_cast<uint16_t>(jfifInfo->width),
			jfifInfo->bitsPerChannel,

            &jfifInfo->quantState,
            &jfifInfo->huffmanState,
            &jfifInfo->scanMap,

            jfifInfo->jpegFmt,
			static_cast<uint16_t>(jfifInfo->restartInterval),
			jfifInfo->exifSections,
			jfifInfo->exifSectionsCount
        );

        Write(fd, header.GetBase(), header.GetSize());
        Write(fd, jfifInfo->h_Bytestream, jfifInfo->bytestreamSize);
        fd.flush();
    }catch(...){
        return FAST_IO_ERROR;
    }

    return FAST_OK;
}

fastStatus_t fastJfifStoreToMemory(
    unsigned char *outputStream,
	unsigned *outputStreamSize,

    fastJfifInfo_t *jfifInfo
){
    try {
		JfifBytestream header;
        AppendHeader(
            header,

			static_cast<uint16_t>(jfifInfo->height),
			static_cast<uint16_t>(jfifInfo->width),
			jfifInfo->bitsPerChannel,

            &jfifInfo->quantState,
            &jfifInfo->huffmanState,
            &jfifInfo->scanMap,

            jfifInfo->jpegFmt,
			static_cast<uint16_t>(jfifInfo->restartInterval),
			jfifInfo->exifSections,
			jfifInfo->exifSectionsCount
        );

		if (header.GetSize() + jfifInfo->bytestreamSize > *outputStreamSize)
			return FAST_INSUFFICIENT_HOST_MEMORY;
		
		memcpy(outputStream, header.GetBase(), header.GetSize());
		memcpy(outputStream + header.GetSize(), jfifInfo->h_Bytestream, jfifInfo->bytestreamSize);
		*outputStreamSize = static_cast<unsigned int>(header.GetSize()) + jfifInfo->bytestreamSize;
    }catch(...){
        return FAST_INTERNAL_ERROR;
    }

    return FAST_OK;
}
