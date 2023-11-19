#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <vortex_filtering/vortex_filtering.hpp>

using std::string;
// using std::endl;

class PDAF {
public:
    using DynModT = vortex::models::CVModel;
    using SensModT = vortex::models::IdentitySensorModel<4, 2>;
    using DynModPtr = std::shared_ptr<DynModT>;
    using SensModPtr = std::shared_ptr<SensModT>;
    using EKF = vortex::filters::EKF<DynModT, SensModT>;
    using Gauss2d = vortex::prob::MultiVarGauss2d;
    using Gauss4d = vortex::prob::MultiVarGauss4d;

    DynModPtr dynamic_model_;
    SensModPtr sensor_model_;
    EKF ekf_;

    PDAF(DynModPtr dynamic_model, SensModPtr sensor_model)
        : dynamic_model_(dynamic_model)
        , sensor_model_(sensor_model)
        , ekf_(dynamic_model, sensor_model)
    {
        std::cout << "Created PDAF class with given models!" << std::endl;
    }

    void predict_next_state(vortex::prob::MultiVarGauss4d x_est, std::vector<Eigen::Vector2d> z_meas, double timestep)
    {
        std::pair<Gauss4d, Gauss2d> x_z_pred = ekf_.predict(x_est, timestep);

        // x_post -> vector
        for (const auto &element : z_meas) {
            // x_post append with:
            ekf_.update(std::get<0>(x_z_pred), std::get<1>(x_z_pred), element);
        }
    }
};

int main(int argc, char** argv)
{
    PDAF pdaf_class();
    return 0;
}