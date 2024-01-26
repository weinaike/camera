#ifndef __LICENSE_INFO__
#define __LICENSE_INFO__

#include <time.h> 
#include <memory.h> 

#include "fastvideo_sdk_define.h"
#include "DecodeError.hpp"
#include "EnumToStringSdk.h"

const char *GetProductName(int productId) {
	switch (productId)
	{
		case 1:
			return "Runtime License";
		case 2:
			return "Developer License";
	}
	return "";
}

const char* GetFeatureName(int featureId) {
	switch (featureId)
	{
		case 1:
			return "Runtime Core functionality";
		case 2:
			return "Core functionality";
	}
	return "";
}

bool LicenseInfoPrinter(void) {
	fastLicenseInfo_t licenseInfo;
	if (!DecodeError(fastLicenseInfo(&licenseInfo))) {
		return false;
	}

	printf("SDK version: %d.%d.%d.%d (build date: %s)\n",
		licenseInfo.sdkVersion[0],
		licenseInfo.sdkVersion[1],
		licenseInfo.sdkVersion[2],
		licenseInfo.sdkVersion[3],
		licenseInfo.buildDate);
	printf("License type: %s.", EnumToString(licenseInfo.licenseType));
	if (licenseInfo.licenseType == FAST_LICENSE_TYPE_TRIAL || licenseInfo.licenseType == FAST_LICENSE_TYPE_DEMO) {
		if (licenseInfo.remainingTrialDays == 0) {
			printf("\nTrial period expired\n\n");
			return false;
		}
		if (licenseInfo.remainingTrialDays / 365.0 < 10)
		{
			printf(" Expired in %d days. ", licenseInfo.remainingTrialDays);
			unsigned day, year, month;
			sscanf(licenseInfo.buildDate, "%d-%d-%d", &year, &month, &day);
			struct tm timeinfo;
			memset(&timeinfo, 0, sizeof(tm));
			timeinfo.tm_year = year - 1900;
			timeinfo.tm_mon = month - 1;
			timeinfo.tm_mday = day;
			time_t expiredDate = mktime(&timeinfo) + licenseInfo.remainingTrialDays * 3600 * 24;
			tm* expiredDateIndo = gmtime(&expiredDate);
			printf("Expired date: %d-%02d-%02d\n\n", expiredDateIndo->tm_year + 1900, expiredDateIndo->tm_mon + 1, expiredDateIndo->tm_mday);
		}
		else
		{
			printf("\nLicense never expires.\n\n");
		}

	} else if (licenseInfo.licenseType == FAST_LICENSE_TYPE_STANDARD_SENSELOCK || licenseInfo.licenseType == FAST_LICENSE_TYPE_STANDARD_GUARDANT) {
		printf("\n\nLicense Provider: %s\n", EnumToString(licenseInfo.licenseProvider));
		if (licenseInfo.licenseProvider == FAST_LICENSE_PROVIDER_NONE)
		{
			printf("License is not found. SDK is disabled.\n");
			return false;
		}

		if (licenseInfo.licenseProvider == FAST_LICENSE_PROVIDER_SENSELOCK_DONGLE)
		{
			printf("\n\nDongle name: %s\n", licenseInfo.senselockInfo.dongleName);
			printf("Dongle license version: %d\n", licenseInfo.senselockInfo.dongleLicenseVersion);
			printf("Dongle id: ");
			for (int i = 0; i < 8; i++) {
				unsigned int t = (unsigned char)licenseInfo.senselockInfo.dongleId[i];
				printf("%02X", t);
			}
			printf("\n\n");
		}
		else if (licenseInfo.licenseProvider == FAST_LICENSE_PROVIDER_GUARDANT_DONGLE || licenseInfo.licenseProvider == FAST_LICENSE_PROVIDER_GUARDANT_SOFT_KEY)
		{
			if (licenseInfo.licenseProvider == FAST_LICENSE_PROVIDER_GUARDANT_DONGLE)
			{
				printf("\n\nDongle Id: %d\n", licenseInfo.guardantInfo.dongleId);
			}
			for (int productId = 0; productId < licenseInfo.guardantInfo.productsCount; productId++)
			{
				printf("Product: %s(%d)\n", GetProductName(licenseInfo.guardantInfo.products[productId].id), licenseInfo.guardantInfo.products[productId].id);
				for (int featureId = 0; featureId < licenseInfo.guardantInfo.products[productId].featuresCount; featureId++)
				{
					printf("Feature: %s(%d)\n", GetFeatureName(licenseInfo.guardantInfo.products[productId].features[featureId].id), licenseInfo.guardantInfo.products[productId].features[featureId].id);
				}
			}
			printf("\n\n");
		}
	} else {
		printf("\n\n");
	}
	return true;
}

bool SDKVersionPrinter(void) {
	fastLicenseInfo_t licenseInfo;
	if (!DecodeError(fastLicenseInfo(&licenseInfo))) {
		return false;
	}

	printf("SDK version: %d.%d.%d.%d\n",
		licenseInfo.sdkVersion[0],
		licenseInfo.sdkVersion[1],
		licenseInfo.sdkVersion[2],
		licenseInfo.sdkVersion[3]
	);
	return true;
}

#endif // __LICENSE_INFO__
