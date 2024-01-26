#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "CollectionFastAllocator.hpp"

std::vector<ConstMemoryManager<FastAllocator>> fastAllocatorConstManagers;
std::vector<VariableMemoryManager<FastAllocator>> fastAllocatorVariableManagers;
std::vector<CollectionMemoryManager<FastAllocator>> fastAllocatorCollectionManagers;
