#include <limits>
#include <cstdio>

int main()
{
	float a = 300.0 / 201;
	float b = 300.0 / 200;

	float relative_error = (b - a) / a;

	if (relative_error < std::numeric_limits<float>::epsilon())
		printf("%.20f and %.20f equal(diff = %.20f, relative_error = %.20f)\n", a, b, b - a, relative_error);
	else
		printf("%.20f and %.20f not equal(diff = %.20f, relative_error = %.20f)\n", a, b, b - a, relative_error);

	printf("epsilon: %.20f\n", std::numeric_limits<float>::epsilon());
}