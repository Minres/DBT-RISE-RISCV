/*******************************************************************************
 * Copyright (C) 2018 MINRES Technologies GmbH
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *******************************************************************************/

#ifndef BLDC_H_
#define BLDC_H_

#include <boost/numeric/odeint.hpp>
namespace odeint = boost::numeric::odeint;

inline double norm_angle(double alpha) {
    double alpha_n = fmod(alpha, M_PI * 2);
    while (alpha_n < 0.) alpha_n += (M_PI * 2);
    return alpha_n;
}

class BLDC {
public:
    struct Config {
        double inertia = 0.0005;      /* aka 'J' in kg/(m^2) */
        double damping = 0.000089;    /* aka 'B' in Nm/(rad/s) */
        double static_friction = 0.0; /* in Nm */
        // double Kv = 0.0042;             /* motor constant in RPM/V */
        double Ke = 0.0042;   /* back emf constant in V/rad/s*/
        double L = 0.0027;    /* Coil inductance in H */
        double M = -0.000069; /* Mutual coil inductance in H */
        double R = 2.875;     /* Coil resistence in Ohm */
        int NbPoles =
            2; /* NbPoles / 2 = Number of pole pairs (you count the permanent magnets on the rotor to get NbPoles) */
    };

    using StateVector = std::array<double, 5>;

    struct State {
        double &theta; /* angle of the rotor */
        double &omega; /* angular speed of the rotor */
        double &ia;    /* phase a current */
        double &ib;    /* phase b current */
        double &ic;    /* phase c current */
        explicit State(StateVector &v)
        : theta(v[0])
        , omega(v[1])
        , ia(v[2])
        , ib(v[3])
        , ic(v[4]) {}
        State(State &&) = delete;
        State(const State &) = delete;
        State &operator=(const State &) = delete;  // Copy assignment operator
        State &operator=(const State &&) = delete; // Move assignment operator
        ~State() {}
        void init() {
            theta = ia = ib = ic = 0;
            omega = 0.;
        }
    };

    explicit BLDC(const Config config);

    virtual ~BLDC();

    void set_input(std::array<double, 3> vin) { this->vin = vin; }

    void run(double dt);

    void printToStream(std::ostream &) const;

    double get_current_time() { return current_time; }

    std::array<double, 7> get_voltages() {
        return std::array<double, 7>{voltages[VA], voltages[VB], voltages[VC], voltages[VCENTER],
                                     voltages[EA], voltages[EB], voltages[EC]};
    }
    const State &getState() { return state; }

    void setLoad(double torque) { torque_load = torque; }

    const double dt = 0.00000001;

protected:
    Config config;
    StateVector stateVector;
    State state;
    std::array<double, 3> vin;
    double current_time = 0.0;
    double torque_load = 0.0001;
    double etorque = 0.0, mtorque = 0.0;
    std::array<double, 7> voltages;
    enum VoltageNames { EA = 0, EB = 1, EC = 2, VA = 3, VB = 4, VC = 5, VCENTER = 6 };
    double calc_bemf_factor(const State &state, double theta);
    void calc_back_emf(const State &state, double theta_e);
    void calc_voltages();
    // ODE part
    // boost::numeric::odeint::runge_kutta4< StateVector > stepper;
    // boost::numeric::odeint::runge_kutta_cash_karp54<StateVector > stepper;
    // using  stepper_type = odeint::runge_kutta_dopri5<StateVector>;
    // using  stepper_type = odeint::runge_kutta_cash_karp54< StateVector>;
    using stepper_type = odeint::runge_kutta_fehlberg78<StateVector>;
    void rotor_dyn(const StateVector &x, StateVector &dxdt, const double t);
};

std::ostream &operator<<(std::ostream &os, const BLDC &bldc);

#endif /* BLDC_H_ */
