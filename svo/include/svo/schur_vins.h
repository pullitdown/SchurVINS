// Copyright 2024 ByteDance and/or its affiliates.
/*
This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef SCHUR_VINS
#define SCHUR_VINS
#include <svo/common/frame.h>
#include <svo/global.h>
#include <boost/math/distributions/chi_squared.hpp>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <set>

namespace schur_vins
{

    class ImuState
    {
    public:
        using Ptr = std::shared_ptr<ImuState>;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ImuState()
        {
        }
        Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat_fej = Eigen::Quaterniond::Identity();
        Eigen::Vector3d pos_fej = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel_fej = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
        double ts = -1;
        int64_t id = 0;
    };

    struct NoiseManager
    {

        /// Gyroscope white noise (rad/s/sqrt(hz))
        double sigma_w = 1.7356915473175545e-03;

        /// Gyroscope white noise covariance
        double sigma_w_2 = pow(sigma_w, 2);

        /// Gyroscope random walk (rad/s^2/sqrt(hz))
        double sigma_wb = 1.3010897230607218e-05;

        /// Gyroscope random walk covariance
        double sigma_wb_2 = pow(sigma_wb, 2);

        /// Accelerometer white noise (m/s^2/sqrt(hz))
        double sigma_a = 1.2623857018616855e-02;

        /// Accelerometer white noise covariance
        double sigma_a_2 = pow(sigma_a, 2);

        /// Accelerometer random walk (m/s^3/sqrt(hz))
        double sigma_ab = 3.2595004354121901e-04;

        /// Accelerometer random walk covariance
        double sigma_ab_2 = pow(sigma_ab, 2);

        /// Nice print function of what parameters we have loaded
    };

    struct UpdaterOptions
    {

        /// What chi-squared multipler we should apply
        int chi2_multipler = 5;

        /// Noise sigma for our raw pixel measurements
        double sigma_pix = 1;

        /// Covariance for our raw pixel measurements
        double sigma_pix_sq = 1;

        /// Nice print function of what parameters we have loaded
    };

    struct IMUDATA
    {

        /// Timestamp of the reading
        double timestamp;

        /// Gyroscope reading, angular velocity (rad/s)
        Eigen::Matrix<double, 3, 1> wm;

        /// Accelerometer reading, linear acceleration (m/s^2)
        Eigen::Matrix<double, 3, 1> am;
    };

    class ZUPTUpdater
    {
    public:
        using Ptr = std::shared_ptr<ImuState>;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ZUPTUpdater(UpdaterOptions &options, NoiseManager &noises, Eigen::Vector3d gravity, double zupt_max_velocity, double zupt_noise_multiplier)
            : _options(options), _noises(noises), _gravity(gravity), _zupt_max_velocity(zupt_max_velocity), _zupt_noise_multiplier(zupt_noise_multiplier)
        {

            // Save our raw pixel noise squared
            _noises.sigma_w_2 = std::pow(_noises.sigma_w, 2);
            _noises.sigma_a_2 = std::pow(_noises.sigma_a, 2);
            _noises.sigma_wb_2 = std::pow(_noises.sigma_wb, 2);
            _noises.sigma_ab_2 = std::pow(_noises.sigma_ab, 2);

            // Initialize the chi squared test table with confidence level 0.95
            // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
            for (int i = 1; i < 1000; i++)
            {
                boost::math::chi_squared chi_squared_dist(i);
                chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
            }
        }

        ZUPTUpdater(Eigen::Vector3d gravity, double zupt_max_velocity, double zupt_noise_multiplier)
            : _gravity(gravity), _zupt_max_velocity(zupt_max_velocity), _zupt_noise_multiplier(zupt_noise_multiplier)
        {

            // Save our raw pixel noise squared
            _noises.sigma_w_2 = std::pow(_noises.sigma_w, 2);
            _noises.sigma_a_2 = std::pow(_noises.sigma_a, 2);
            _noises.sigma_wb_2 = std::pow(_noises.sigma_wb, 2);
            _noises.sigma_ab_2 = std::pow(_noises.sigma_ab, 2);

            // Initialize the chi squared test table with confidence level 0.95
            // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
            for (int i = 1; i < 1000; i++)
            {
                boost::math::chi_squared chi_squared_dist(i);
                chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
            }
        }

        inline void feed_imus(const svo::ImuMeasurements &imus)
        {

            imu_data.clear();
            // Create our imu data object

            for (auto &a : imus)
            {
                IMUDATA data;
                data.timestamp = a.timestamp_;
                data.wm = a.angular_velocity_;
                data.am = a.linear_acceleration_;

                // Append it to our vector
                imu_data.emplace_back(data);
            }
        }

        /**
         * @brief Stores incoming inertial readings
         * @param timestamp Timestamp of imu reading
         * @param wm Gyro angular velocity reading
         * @param am Accelerometer linear acceleration reading
         */

        inline void feed_imu(const svo::ImuMeasurement &imu)
        {

            // Create our imu data object
            IMUDATA data;
            data.timestamp = imu.timestamp_;
            data.wm = imu.angular_velocity_;
            data.am = imu.linear_acceleration_;

            // Append it to our vector
            imu_data.emplace_back(data);

            // Sort our imu data (handles any out of order measurements)
            // std::sort(imu_data.begin(), imu_data.end(), [](const IMUDATA i, const IMUDATA j) {
            //    return i.timestamp < j.timestamp;
            //});

            // Loop through and delete imu messages that are older then 20 seconds
            // TODO: we should probably have more elegant logic then this
            // TODO: but this prevents unbounded memory growth and slow prop with high freq imu
            auto it0 = imu_data.begin();
            while (it0 != imu_data.end())
            {
                if (data.timestamp - (*it0).timestamp > 60)
                {
                    it0 = imu_data.erase(it0);
                }
                else
                {
                    return;
                }
            }
        }

        Eigen::Quaterniond updateQuaternionWithDelta(const Eigen::Quaterniond &q, const Eigen::Vector3d &delta_theta)
        {
            // 计算增量四元数
            Eigen::Quaterniond dq;
            Eigen::Vector3d half_delta_theta = 0.5 * delta_theta;
            double half_angle = half_delta_theta.norm();

            if (half_angle > 1e-10)
            {
                dq.w() = std::cos(half_angle);
                dq.vec() = (half_delta_theta / half_angle) * std::sin(half_angle);
            }
            else
            {
                // 小角度近似
                dq.w() = 1.0;
                dq.vec() = half_delta_theta;
            }

            // 更新四元数
            Eigen::Quaterniond q_new = dq * q;
            return q_new.normalized(); // 确保四元数归一化
        }

        /**
         * @brief Will first detect if the system is zero velocity, then will update.
         * @param state State of the filter
         * @param timestamp Next camera timestamp we want to see if we should propagate to.
         * @return True if the system is currently at zero velocity
         */
        bool try_update(Ptr state_ptr, Eigen::MatrixXd &cov, double timestamp);

        inline void set_last_prop_time_offset(double time_offset)
        {
            last_prop_time_offset = time_offset;
        }

        IMUDATA interpolate_data(const IMUDATA imu_1, const IMUDATA imu_2, double timestamp)
        {
            // time-distance lambda
            double lambda = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
            // cout << "lambda - " << lambda << endl;
            //  interpolate between the two times
            IMUDATA data;
            data.timestamp = timestamp;
            data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
            data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
            return data;
        };

        std::vector<IMUDATA> select_imu_readings(const std::vector<IMUDATA> &imu_data, double time0, double time1)
        {

            VLOG(40)<<"imu_readings oldest timestamp and newest timestamp"<< std::setprecision(16)<<imu_data.at(0).timestamp<<","<< std::setprecision(16)<<imu_data.at(imu_data.size()-1).timestamp<< \
            " need timestamp"<< std::setprecision(16)<<time0<<","<< std::setprecision(16)<<time1;
            // Our vector imu readings
            std::vector<IMUDATA> prop_data;

            // Ensure we have some measurements in the first place!
            if (imu_data.empty())
            {
                return prop_data;
            }

            // Loop through and find all the needed measurements to propagate with
            // Note we split measurements based on the given state time, and the update timestamp
            for (size_t i = 0; i < imu_data.size() - 1; i++)
            {

                // START OF THE INTEGRATION PERIOD
                // If the next timestamp is greater then our current state time
                // And the current is not greater then it yet...
                // Then we should "split" our current IMU measurement
                if (imu_data.at(i + 1).timestamp > time0 && imu_data.at(i).timestamp < time0)
                {
                    IMUDATA data = interpolate_data(imu_data.at(i), imu_data.at(i + 1), time0);
                    prop_data.push_back(data);
                    // printf("propagation #%d = CASE 1 = %.3f => %.3f\n", (int)i,data.timestamp-prop_data.at(0).timestamp,time0-prop_data.at(0).timestamp);
                    continue;
                }

                // MIDDLE OF INTEGRATION PERIOD
                // If our imu measurement is right in the middle of our propagation period
                // Then we should just append the whole measurement time to our propagation vector
                if (imu_data.at(i).timestamp >= time0 && imu_data.at(i + 1).timestamp <= time1)
                {
                    prop_data.push_back(imu_data.at(i));
                    // printf("propagation #%d = CASE 2 = %.3f\n",(int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp);
                    continue;
                }

                // END OF THE INTEGRATION PERIOD
                // If the current timestamp is greater then our update time
                // We should just "split" the NEXT IMU measurement to the update time,
                // NOTE: we add the current time, and then the time at the end of the interval (so we can get a dt)
                // NOTE: we also break out of this loop, as this is the last IMU measurement we need!
                if (imu_data.at(i + 1).timestamp > time1)
                {
                    // If we have a very low frequency IMU then, we could have only recorded the first integration (i.e. case 1) and nothing else
                    // In this case, both the current IMU measurement and the next is greater than the desired intepolation, thus we should just cut the current at the desired time
                    // Else, we have hit CASE2 and this IMU measurement is not past the desired propagation time, thus add the whole IMU reading
                    if (imu_data.at(i).timestamp > time1)
                    {
                        IMUDATA data = interpolate_data(imu_data.at(i - 1), imu_data.at(i), time1);
                        prop_data.push_back(data);
                        // printf("propagation #%d = CASE 3.1 = %.3f => %.3f\n", (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
                    }
                    else
                    {
                        prop_data.push_back(imu_data.at(i));
                        // printf("propagation #%d = CASE 3.2 = %.3f => %.3f\n", (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
                    }
                    // If the added IMU message doesn't end exactly at the camera time
                    // Then we need to add another one that is right at the ending time
                    if (prop_data.at(prop_data.size() - 1).timestamp != time1)
                    {
                        IMUDATA data = interpolate_data(imu_data.at(i), imu_data.at(i + 1), time1);
                        prop_data.push_back(data);
                        // printf("propagation #%d = CASE 3.3 = %.3f => %.3f\n", (int)i,data.timestamp-prop_data.at(0).timestamp,data.timestamp-time0);
                    }
                    break;
                }
            }

            // Check that we have at least one measurement to propagate with
            if (prop_data.empty())
            {
                return prop_data;
            }

            // If we did not reach the whole integration period (i.e., the last inertial measurement we have is smaller then the time we want to reach)
            // Then we should just "stretch" the last measurement to be the whole period (case 3 in the above loop)
            // if(time1-imu_data.at(imu_data.size()-1).timestamp > 1e-3) {
            //    printf(YELLOW " select_imu_readings(): Missing inertial measurements to propagate with (%.6f sec missing). IMU-CAMERA are likely messed up!!!\n" RESET, (time1-imu_data.at(imu_data.size()-1).timestamp));
            //    return prop_data;
            //}

            // Loop through and ensure we do not have an zero dt values
            // This would cause the noise covariance to be Infinity
            for (size_t i = 0; i < prop_data.size() - 1; i++)
            {
                if (std::abs(prop_data.at(i + 1).timestamp - prop_data.at(i).timestamp) < 1e-12)
                {
                    prop_data.erase(prop_data.begin() + i);
                    i--;
                }
            }

            // Check that we have at least one measurement to propagate with
            if (prop_data.size() < 2)
            {
                return prop_data;
            }

            // Success :D
            return prop_data;
        };

    protected:
        /// Options used during update (chi2 multiplier)
        UpdaterOptions _options;

        /// Container for the imu noise values
        NoiseManager _noises;

        /// Gravity vector
        Eigen::Matrix<double, 3, 1> _gravity;

        /// Max velocity (m/s) that we should consider a zupt with
        double _zupt_max_velocity = 1.0;

        /// Multiplier of our IMU noise matrix (default should be 1.0)
        double _zupt_noise_multiplier = 1.0;

        /// Chi squared 95th percentile table (lookup would be size of residual)
        std::map<int, double> chi_squared_table;

        /// Our history of IMU messages (time, angular, linear)
        std::vector<IMUDATA> imu_data;

        /// Estimate for time offset at last propagation time
        double last_prop_time_offset = 0;
        bool have_last_prop_time_offset = false;
    };

    class SchurVINS
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        SchurVINS();

        void InitImuModel(double acc_n, double acc_w, double gyr_n, double gyr_w);
        void InitExtrinsic(const svo::CameraBundle::Ptr camera_bundle);
        void InitCov();
        void InitMaxState(int val);
        void InitFocalLength(double val);
        void InitObsStddev(double _obs_dev);
        void InitChi2(double chi2_rate);

        void SetKeyframe(const bool _is_keyframe);
        void Forward(const svo::FrameBundle::Ptr frame_bundle);
        int Backward(const svo::FrameBundle::Ptr frame_bundle);
        bool try_zerovelocity_update(double timestamp);

        bool StructureOptimize(const svo::PointPtr &optimize_points);
        void GetImuState(Eigen::Quaterniond &quat, Eigen::Vector3d &pos, Eigen::Vector3d &vel, Eigen::Vector3d &gyr, double &time)
        {
            quat = curr_state->quat;
            pos = curr_state->pos;
            vel = curr_state->vel;
            time = curr_state->ts;
            gyr = curr_state->gyr;
        };
        void SetImuState(const Eigen::Quaterniond &quat, const Eigen::Vector3d &pos)
        {
            curr_state->quat = quat;
            curr_state->pos = pos;
        };

        void set_imu_delay_time(double time)
        {
            delay_imu_cam = time;
        }

        void feed_zupt_imu_datas(svo::ImuMeasurements &a)
        {
            zuptupdater_->feed_imus(a);
        }

    private:
        void InitState(double _ts, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr);
        void AugmentState(const svo::FrameBundle::Ptr frame_bundle);
        void PredictionState(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyro, double dt);
        void Prediction(double _ts, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr);
        void Management(int index);
        void ManageLocalMap();
        bool KeyframeSelect();
        void StateUpdate(const Eigen::MatrixXd &hessian, const Eigen::VectorXd &gradient);
        void StateUpdate2(const Eigen::MatrixXd &hessian, const Eigen::VectorXd &gradient);
        // bool Chi2Check(const Eigen::MatrixXd& j, const Eigen::VectorXd& r, int dof);
        void StateCorrection(const Eigen::MatrixXd &K, const Eigen::MatrixXd &J, const Eigen::VectorXd &dX,
                             const Eigen::MatrixXd &R);
        void RegisterPoints(const svo::FrameBundle::Ptr &frame_bundle);

        void Solve3();
        int RemoveOutliers(const svo::FrameBundle::Ptr frame_bundle);
        int RemovePointOutliers();

    private:
        std::mutex msg_queue_mtx;
        std::condition_variable con;
        svo::StateMap states_map;
        ImuState::Ptr curr_state = nullptr;
        svo::CameraPtr cam0_param = nullptr, cam1_param = nullptr;
        svo::Transformation T_imu_cam0, T_imu_cam1;
        bool stereo = false;
        svo::LocalPointMap local_pts;

        // imu prediction
        Eigen::Vector3d prev_acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d prev_gyr = Eigen::Vector3d::Zero();
        double prev_imu_ts = -1;

        Matrix12d imu_noise = Matrix12d::Zero();
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(15, 15);
        int64_t id_creator = 0;
        Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);
        Eigen::VectorXd dsw_;
        std::set<svo::PointPtr> schur_pts_;
        std::shared_ptr<ZUPTUpdater> zuptupdater_;

        int state_max = 4;
        int curr_fts_cnt = 0;
        double focal_length = 1000;
        bool zupt_valid = false;
        double obs_dev = 1;
        double obs_invdev = 1;
        double huberA = 1.5;
        double huberB = huberA * huberA;
        double delay_imu_cam = 0;
        std::vector<double> chi_square_lut;
    };

    namespace Utility
    {
        inline Eigen::Matrix3d SkewMatrix(const Eigen::Vector3d &w)
        {
            Eigen::Matrix3d mat;
            mat << 0, -w.coeff(2), w.coeff(1), w.coeff(2), 0, -w.coeff(0), -w.coeff(1), w.coeff(0), 0;
            return mat;
        }
    } // namespace Utility
} // namespace schur_vins
#endif