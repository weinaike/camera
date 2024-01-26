
#include "alignment.hpp"

unsigned GetCompactedFfcWidth(const unsigned width) {
	return uDivUp(width, 4u) + 1;
}

unsigned GetCompactedFfcHeight(const unsigned height) {
	return uDivUp(height, 4u) + 1;
}

unsigned GetFfcCoeffs(const unsigned width, const unsigned height) {
	return
		GetCompactedFfcWidth(width) * GetCompactedFfcHeight(height);
}
