#include "theory.h"
#include "constants.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

namespace theory { // Wrap content in theory namespace

// Unit conventions: radius in mm, power in W, result irradiance in W/mm²

TheoryResult TheoryCalculator::calculate(
    const TheoreticalIntegratingSphere& sphere,
    float incident_power
) {
    TheoryResult result;

    const float radius = sphere.get_radius();
    double effective_reflectance = sphere.get_effective_wall_reflectance();

    // Calculate total surface area (mm²)
    result.sphere_area = sphere.get_total_sphere_area();

    // Average Irradiance (Goebel Formula) (W/mm²)
    // E = P_incident / (A_sphere * (1 - ρ_eff))
    if (effective_reflectance >= 1.0) {
        result.avg_irradiance = INFINITY;
    } else {
        result.avg_irradiance = incident_power /
                                (result.sphere_area * (1.0 - effective_reflectance));
    }

    result.total_flux_in_sphere = incident_power / (1.0 - effective_reflectance);

    return result;
}

// printComparison is removed as it's a display function, not a calculation,
// and can be handled at a higher level (e.g., in tests or Python examples).


} // namespace theory