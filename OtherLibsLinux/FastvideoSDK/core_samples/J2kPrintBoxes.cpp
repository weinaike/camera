
#include "J2kPrintBoxes.h"

#include <cstdio>
#include <cstring>

int PrintJ2kGML(fastJ2kImageInfo_t* info)
{
    const char* label = "gml.root-instance";
    size_t labelLength = strlen(label);
    for (unsigned i = 0; i < info->asocBoxesCount; i++)
    {
        if (info->asoc[i].labelCount > 0 && strncmp(reinterpret_cast<const char*>(info->asoc[i].labels[0]), label, labelLength) == 0)
        {
            printf("%s\n", info->asoc[i].XMLs[0]);
            return 1;
        }
    }
    return 0;
}
void PrintUUID(fastJ2kUuid_t uuid)
{
    unsigned char* ptr = (unsigned char*)(&uuid);
    for (int j = 0; j < 16; j++)
        printf("%X ", ptr[j]);
}
void PrintJ2kReaderRequirementBox(fastJ2kImageInfo_t* info)
{
    if (info->containsRreqBox)
    {
        unsigned i;
        printf("Contains rreq box with %d standard flags, %d vendor features\n", info->rreq.standardFlagsCount, info->rreq.vendorFeatureCount);
        if (info->rreq.standardFlagsCount > 0)
        {
            printf("\tstandard flags: ");
            for (i = 0; i < info->rreq.standardFlagsCount; i++)
                printf("%d ", info->rreq.standardFlags[i]);
            printf("\n");
            printf("\tstandard masks: ");
            for (i = 0; i < info->rreq.standardFlagsCount; i++)
                printf("%llu ", info->rreq.standardMasks[i]);
            printf("\n");
        }
        if (info->rreq.vendorFeatureCount > 0)
        {
            printf("\tvendor features:\n");
            for (i = 0; i < info->rreq.vendorFeatureCount; i++)
            {
                printf("\t\t");
                PrintUUID(info->rreq.vendorFeatures[i]);
                printf("\n");
            }
            printf("\n");
            printf("\tvendor masks: ");
            for (i = 0; i < info->rreq.vendorFeatureCount; i++)
                printf("%llu ", info->rreq.vendorMasks[i]);
            printf("\n");
        }
        if (info->asocBoxesCount > 0) printf("\n");
    }
}
void PrintTabs(int count)
{
    for (int i = 0; i < count; i++) putchar('\t');
}
void PrintJ2kAsocBox(fastJ2kAssociationBox_t* asoc, int tab_count)
{
    unsigned i;
    PrintTabs(tab_count);
    printf("asoc contains %d labels, %d XML, %d children\n", asoc->labelCount, asoc->xmlCount, asoc->childrenCount);
    for (i = 0; i < asoc->labelCount; i++)
    {
        PrintTabs(tab_count + 1);
        printf("label: %s\n", asoc->labels[i]);
    }
    for (i = 0; i < asoc->xmlCount; i++)
    {
        PrintTabs(tab_count + 1);
        if (asoc->xmlLengths[i] < 13)
        printf("XML: %s\n", asoc->XMLs[i]);
        else
        {
            char s[16];
            snprintf(s, 13, "%s", asoc->XMLs[i]);
            s[12] = s[13] = s[14] = '.'; s[15] = 0;
            printf("XML: %s\n", s);
        }
    }
    for (i = 0; i < asoc->childrenCount; i++)
    {
        PrintJ2kAsocBox(asoc->children[i], tab_count + 1);
    }
}
void PrintJ2kAsocBoxes(fastJ2kImageInfo_t* info)
{
    printf("Contains: %d asoc boxes\n", info->asocBoxesCount);
    for (unsigned i = 0; i < info->asocBoxesCount; i++)
        if (!info->asoc[i].isChild)
        {
            PrintJ2kAsocBox(&info->asoc[i], 1);
        }
}
void PrintJ2kUUIDboxes(fastJ2kImageInfo_t* info)
{
    unsigned i;
    printf("Contains: %d uuid %d uinf boxes\n", info->uuidBoxesCount, info->containsUuidInfoBox ? 1 : 0);
    for (i = 0; i < info->uuidBoxesCount; i++)
    {
        printf("\tUUID %d: ", i);
        PrintUUID(info->uuidBoxes[i].id);
        printf("\t data (length = %d)\n", info->uuidBoxes[i].dataLength);
    }
    if (info->containsUuidInfoBox)
    {
        if (info->uuidBoxesCount > 0) printf("\n");
        printf("\tUUID Info: %d IDs, url version %d, url flags %d\n", info->uuidInfo.idCount, info->uuidInfo.urlVersion, info->uuidInfo.urlFlags);
        printf("\t\turl = %s\n", info->uuidInfo.url);
        for (i = 0; i < info->uuidInfo.idCount; i++)
        {
            printf("\tUUID %d: ", i);
            PrintUUID(info->uuidInfo.IDs[i]);
            printf("\n");
        }
    }
}
