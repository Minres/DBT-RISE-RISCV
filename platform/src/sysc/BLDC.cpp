/*
 * BLDC.cpp
 *
 *  Created on: 26.06.2018
 *      Author: eyck
 */

#include "sysc/top/BLDC.h"

// implementation according to Modeling of BLDC Motor with Ideal Back-EMF for Automotive Applications
// Proceedings of the World Congress on Engineering 2011 Vol II WCE 2011, July 6 - 8, 2011, London, U.K.
BLDC::BLDC(const Config config)
: config(config)
, stateVector({{0.0, 0.0, 0.0, 0.0, 0.0}})
, state(stateVector)
, vin({{0.0, 0.0, 0.0}})
, voltages({{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}})
{
    state.init();
}

BLDC::~BLDC() {

}

double BLDC::calc_bemf_factor(const State& x, double theta){
    if(theta>=0 && theta < 2./3.*M_PI){
        return 1;
    } else if(theta>=2./3.*M_PI && theta < M_PI){
        return 1-6/M_PI*(theta-2./3.*M_PI);
    } else if(theta>=M_PI && theta < 5./3. * M_PI){
        return -1;
    } else if(theta>=5./3. * M_PI && theta < 2. * M_PI){
        return -1+6/M_PI*(theta-5./3.*M_PI);
    } else {
        fprintf(stderr, "ERROR: angle out of bounds can not calculate bemf %f\n", theta);
        throw std::runtime_error("angle out of bounds can not calculate bemf");
    }
}

void BLDC::calc_back_emf(const State& state, double theta_e) {
    double max_bemf = config.Ke * state.omega;
    voltages[EA] = max_bemf*calc_bemf_factor(state, norm_angle(theta_e));
    voltages[EB] = max_bemf*calc_bemf_factor(state, norm_angle(theta_e + M_PI * (2. / 3.)));
    voltages[EC] = max_bemf*calc_bemf_factor(state, norm_angle(theta_e + M_PI * (4. / 3.)));
}


void BLDC::calc_voltages(){
    const double NaN = nan("");
    /* Check which phases are excited. */
    bool pa = isnan(vin[0])?false:true;
    bool pb = isnan(vin[1])?false:true;
    bool pc = isnan(vin[2])?false:true;

    if (pa && pb && pc) {
        voltages[VA] = vin[0];
        voltages[VB] = vin[1];
        voltages[VC] = vin[2];
        voltages[VCENTER] = (voltages[VA] + voltages[VB] + voltages[VC] - voltages[EA] - voltages[EB] - voltages[EC]) / 3.;
    } else if (pa && pb) {
        voltages[VA] = vin[0];
        voltages[VB] = vin[1];
        voltages[VCENTER] = (voltages[VA] + voltages[VB] - voltages[EA] - voltages[EB]) / 2.;
        voltages[VC] = voltages[EC] + voltages[VCENTER];
    } else if (pa && pc) {
        voltages[VA] = vin[0];
        voltages[VC] = vin[2];
        voltages[VCENTER] = (voltages[VA] + voltages[VC] - voltages[EA] - voltages[EC]) / 2.;
        voltages[VB] = voltages[EB] + voltages[VCENTER];
    } else if (pb && pc) {
        voltages[VB] = vin[1];
        voltages[VC] = vin[2];
        voltages[VCENTER] = (voltages[VB] + voltages[VC] - voltages[EB] - voltages[EC]) / 2.;
        voltages[VA] = voltages[EA] + voltages[VCENTER];
    } else if (pa) {
        voltages[VA] = vin[0];
        voltages[VCENTER] = (voltages[VA] - voltages[EA]);
        voltages[VB] = voltages[EB] + voltages[VCENTER];
        voltages[VC] = voltages[EC] + voltages[VCENTER];
    } else if (pb) {
        voltages[VB] = vin[1];
        voltages[VCENTER] = (voltages[VB] - voltages[EB]);
        voltages[VA] = voltages[EA] + voltages[VCENTER];
        voltages[VC] = voltages[EC] + voltages[VCENTER];
    } else if (pc) {
        voltages[VC] = vin[0];
        voltages[VCENTER] = (voltages[VC] - voltages[EC]);
        voltages[VA] = voltages[EA] + voltages[VCENTER];
        voltages[VB] = voltages[EB] + voltages[VCENTER];
    } else {
        voltages[VA] = voltages[EA];
        voltages[VB] = voltages[EB];
        voltages[VC] = voltages[EC];
        voltages[VCENTER] = 0;
    }
}

void BLDC::printToStream(std::ostream& os) const {
    os<<state.omega<<";"<<state.theta<<";"
            <<state.ia<<";"<<state.ib<<";"<<state.ic<<";"
            <<voltages[VA]<<";"<<voltages[VB]<<";"<<voltages[VC]<<";"
            <<voltages[EA]<<";"<<voltages[EB]<<";"<<voltages[EC]<<";"<<voltages[VCENTER]<<";"
            <<vin[0]<<";"<<vin[1]<<";"<<vin[2]<<";"<<etorque;
}

void BLDC::rotor_dyn(const StateVector& x_, StateVector& dxdt_, const double t) {
    const State x(const_cast<StateVector&>(x_));
    State dxdt(dxdt_);
    double theta_e = state.theta * (config.NbPoles / 2.);
    /* Calculate backemf voltages. */
    calc_back_emf(x, theta_e);
    /* Calculate voltages. */
    calc_voltages();
    /* Electromagnetic torque. */
//    if (x.omega == 0) {
//        printf("ERROR: input state vector omega equals 0!!!\n");
//        throw std::runtime_error("input state vector omega equals 0");
//    }
    /* electrical torque */
    //etorque = ((voltages[EA] * x.ia) + (voltages[EB] * x.ib) + (voltages[EC] * x.ic)) / x.omega;
    // which is equivalent to:
    etorque = config.Ke*(
            x.ia * (calc_bemf_factor(state, norm_angle(theta_e))) +
            x.ib * (calc_bemf_factor(state, norm_angle(theta_e + M_PI * (2. / 3.)))) +
            x.ic * (calc_bemf_factor(state, norm_angle(theta_e + M_PI * (4. / 3.))))
            );
    /* Mechanical torque. */
    mtorque = ((etorque * (config.NbPoles / 2)) - (config.damping * x.omega) - torque_load);

    if ((mtorque > 0) && (mtorque <= config.static_friction)) {
        mtorque = 0;
    } else if (mtorque > config.static_friction) {
        mtorque -= config.static_friction;
    } else if ((mtorque < 0) && (mtorque >= -(config.static_friction))) {
        mtorque = 0;
    } else if (mtorque < -(config.static_friction)) {
        mtorque += config.static_friction;
    }
    /* Position of the rotor */
    dxdt.theta = x.omega;
    /* Acceleration of the rotor. (omega_dot) */
    // a=M/J with M->torque, J->Inertia, a->angular acceleration
    dxdt.omega = mtorque / config.inertia;
    /* Calculate dot currents. */
    dxdt.ia = (voltages[VA] - (config.R * x.ia) - voltages[EA] - voltages[VCENTER]) / (config.L - config.M);
    dxdt.ib = (voltages[VB] - (config.R * x.ib) - voltages[EB] - voltages[VCENTER]) / (config.L - config.M);
    dxdt.ic = (voltages[VC] - (config.R * x.ic) - voltages[EC] - voltages[VCENTER]) / (config.L - config.M);
}

void BLDC::run(double incr) {
    if(dt>incr) throw std::runtime_error("incr needs to be larger than dt");
    double next_time = current_time+incr;
    odeint::integrate_adaptive(make_controlled( 1.0e-10 , 1.0e-6 , stepper_type() ),
            [this]( const StateVector &x , StateVector &dxdt , double t ) {this->rotor_dyn(x, dxdt,t);},
            stateVector, current_time, next_time, dt);
    current_time=next_time;
    state.theta=norm_angle(state.theta);
}

std::ostream& operator <<(std::ostream& os, const BLDC& bldc) {
    bldc.printToStream(os);
    return os;
}
