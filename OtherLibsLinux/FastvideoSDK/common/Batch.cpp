#include "Batch.hpp"

template<typename T>
Batch<T>::Batch(int batchSize) {
	this->batchSize = batchSize;
	this->filledItem = batchSize;

	values = new T[batchSize];
	isFree = true;
}

template<typename T>
Batch<T>::~Batch() {
	delete[] values;
}

template<typename T>
T* Batch<T>::At(int i) {
	if (i < batchSize)
		return &values[i];
	return nullptr;
}

template<typename T>
unsigned int Batch<T>::GetSize() const {
	return batchSize;
}

template<typename T>
unsigned int Batch<T>::GetFilledItem() const {
	return this->filledItem;
}

template<typename T>
void Batch<T>::SetFilltedItem(unsigned int count) {
	this->filledItem = count;
}
