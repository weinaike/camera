#pragma once

inline const char* PluralSuffix(unsigned value)
{
	if (value > 1)
		return "s";
	return "";
}
