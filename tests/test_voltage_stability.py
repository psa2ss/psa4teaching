"""Voltage stability tests."""
import numpy as np; import copy
from psa4teaching.models import Bus, BusType, Line, Generator, Load
from psa4teaching.stability import compute_pv_curve, compute_qv_curve

class TestPVCurve:
    def test_simple_system(self):
        buses = [Bus(1,'G1',BusType.SLACK,V_specified=1.05),
                 Bus(2,'Ld',BusType.PQ,P_specified=-0.5,Q_specified=-0.2)]
        lines = [Line(1,2,R=0.02,X=0.1,B=0.02)]
        gens = [Generator(bus=1,Xd=1.8,H=5.0)]
        loads = [Load(bus=2,P0=0.5,Q0=0.2)]
        pv = compute_pv_curve(copy.deepcopy(buses),lines,[],gens,loads,
                              lambda_max=2,n_points=20)
        assert pv.converged
        assert pv.critical_lambda > 0
        assert pv.V_curve.shape[0] > 0

    def test_voltage_decreases(self):
        buses = [Bus(1,'G1',BusType.SLACK,V_specified=1.05),
                 Bus(2,'Ld',BusType.PQ,P_specified=-0.8,Q_specified=-0.3)]
        lines = [Line(1,2,R=0.02,X=0.1,B=0.02)]
        gens = [Generator(bus=1,Xd=1.8,H=5.0)]
        loads = [Load(bus=2,P0=0.8,Q0=0.3)]
        pv = compute_pv_curve(copy.deepcopy(buses),lines,[],gens,loads,
                              lambda_max=1,n_points=10)
        assert pv.V_curve[-1, 1] < pv.V_curve[0, 1], "Voltage should decrease with load"

    def test_nose_point_valid(self):
        buses = [Bus(1,'G1',BusType.SLACK,V_specified=1.05),
                 Bus(2,'Ld',BusType.PQ,P_specified=-0.8,Q_specified=-0.3)]
        lines = [Line(1,2,R=0.02,X=0.1,B=0.02)]
        gens = [Generator(bus=1,Xd=1.8,H=5.0)]
        loads = [Load(bus=2,P0=0.8,Q0=0.3)]
        pv = compute_pv_curve(copy.deepcopy(buses),lines,[],gens,loads,
                              lambda_max=3,n_points=30)
        assert 0 <= pv.nose_point_index < 30

class TestQVCurve:
    def test_simple_qv(self):
        buses = [Bus(1,'G1',BusType.SLACK,V_specified=1.05),
                 Bus(2,'Ld',BusType.PQ,P_specified=-0.8,Q_specified=-0.3)]
        lines = [Line(1,2,R=0.02,X=0.1,B=0.02)]
        gens = [Generator(bus=1,Xd=1.8,H=5.0)]
        loads = [Load(bus=2,P0=0.8,Q0=0.3)]
        qv = compute_qv_curve(copy.deepcopy(buses),lines,[],gens,loads,
                              target_bus=2,V_range=(0.4,1.2),n_points=20)
        assert qv.converged
        assert len(qv.V_values) == 20

    def test_q_min_exists(self):
        buses = [Bus(1,'G1',BusType.SLACK,V_specified=1.05),
                 Bus(2,'Ld',BusType.PQ,P_specified=-0.6,Q_specified=-0.2)]
        lines = [Line(1,2,R=0.02,X=0.1,B=0.02)]
        gens = [Generator(bus=1,Xd=1.8,H=5.0)]
        loads = [Load(bus=2,P0=0.6,Q0=0.2)]
        qv = compute_qv_curve(copy.deepcopy(buses),lines,[],gens,loads,
                              target_bus=2,V_range=(0.3,1.2),n_points=30)
        assert qv.Q_min < 0, "Q_min should be negative (reactive injection needed)"
