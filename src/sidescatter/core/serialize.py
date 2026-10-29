"""Plain-JSON (de)serialisation of FlowKit gates and transforms for workspace files."""

import flowkit as fk
import numpy as np

from .scales import BiexTransform, SymLogTransform

_TRANSFORMS = {
    "linear": (fk.transforms.LinearTransform, ("param_t", "param_a")),
    "log": (fk.transforms.LogTransform, ("param_t", "param_m")),
    "logicle": (fk.transforms.LogicleTransform, ("param_t", "param_w", "param_m", "param_a")),
    "asinh": (fk.transforms.AsinhTransform, ("param_t", "param_m", "param_a")),
    "hyperlog": (fk.transforms.HyperlogTransform, ("param_t", "param_w", "param_m", "param_a")),
    "flowjo_biex": (fk.transforms.WSPBiexTransform, ("negative", "width", "positive", "max_value")),
    "biex": (BiexTransform, ("negative", "width", "positive", "max_value")),
    "neglog": (SymLogTransform, ("param_t", "param_n", "param_c")),
}


def _num(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_num(v) for v in x]
    return float(x)


def transform_to_dict(t):
    for kind, (cls, params) in _TRANSFORMS.items():
        if type(t) is cls:
            return {"type": kind, **{p: _num(getattr(t, p)) for p in params}}
    raise TypeError(f"cannot save transform {type(t).__name__}")


def transform_from_dict(d):
    cls, params = _TRANSFORMS[d["type"]]
    return cls(*[d[p] for p in params])


def _dim_to_dict(dim):
    return {"id": dim.id, "compensation_ref": dim.compensation_ref,
            "transformation_ref": dim.transformation_ref,
            "min": _num(dim.min), "max": _num(dim.max)}


def _dim_from_dict(d):
    return fk.Dimension(d["id"], compensation_ref=d["compensation_ref"] or "uncompensated",
                        transformation_ref=d["transformation_ref"],
                        range_min=d["min"], range_max=d["max"])


def gate_to_dict(gate, parent_path):
    d = {"name": gate.gate_name, "parent_path": list(parent_path), "type": gate.gate_type,
         "use_complement": bool(getattr(gate, "use_complement", False))}
    if isinstance(gate, fk.gates.RectangleGate):
        d["dimensions"] = [_dim_to_dict(x) for x in gate.dimensions]
    elif isinstance(gate, fk.gates.PolygonGate):
        d["dimensions"] = [_dim_to_dict(x) for x in gate.dimensions]
        d["vertices"] = _num(gate.vertices)
    elif isinstance(gate, fk.gates.EllipsoidGate):
        d["dimensions"] = [_dim_to_dict(x) for x in gate.dimensions]
        d["coordinates"] = _num(gate.coordinates)
        d["covariance_matrix"] = _num(gate.covariance_matrix)
        d["distance_square"] = _num(gate.distance_square)
    elif isinstance(gate, fk.gates.QuadrantGate):
        d["dividers"] = [{"id": v.id, "dimension_ref": v.dimension_ref,
                          "compensation_ref": v.compensation_ref,
                          "transformation_ref": v.transformation_ref, "values": _num(v.values)}
                         for v in gate.dimensions]
        quads = gate.quadrants.values() if isinstance(gate.quadrants, dict) else gate.quadrants
        d["quadrants"] = [{"id": q.id, "divider_refs": list(q.divider_refs),
                           "ranges": {k: _num(v) for k, v in q._divider_ranges.items()}}
                          for q in quads]
    elif isinstance(gate, fk.gates.BooleanGate):
        d["bool_type"] = gate.type
        d["gate_refs"] = [{"ref": r["ref"], "path": list(r["path"]),
                           "complement": bool(r.get("complement", False))} for r in gate.gate_refs]
    else:
        raise TypeError(f"cannot save gate type {gate.gate_type}")
    return d


def gate_from_dict(d):
    t = d["type"]
    comp = d.get("use_complement", False)
    if t == "RectangleGate":
        return fk.gates.RectangleGate(d["name"], [_dim_from_dict(x) for x in d["dimensions"]],
                                      use_complement=comp)
    if t == "PolygonGate":
        return fk.gates.PolygonGate(d["name"], [_dim_from_dict(x) for x in d["dimensions"]],
                                    d["vertices"], use_complement=comp)
    if t == "EllipsoidGate":
        return fk.gates.EllipsoidGate(d["name"], [_dim_from_dict(x) for x in d["dimensions"]],
                                      d["coordinates"], d["covariance_matrix"], d["distance_square"])
    if t == "QuadrantGate":
        dividers = [fk.QuadrantDivider(v["id"], v["dimension_ref"], v["compensation_ref"],
                                       v["values"], v["transformation_ref"]) for v in d["dividers"]]
        quads = [fk.gates.Quadrant(q["id"], q["divider_refs"],
                                   [tuple(q["ranges"][r]) for r in q["divider_refs"]])
                 for q in d["quadrants"]]
        return fk.gates.QuadrantGate(d["name"], dividers, quads)
    if t == "BooleanGate":
        refs = [{"ref": r["ref"], "path": tuple(r["path"]), "complement": r["complement"]}
                for r in d["gate_refs"]]
        return fk.gates.BooleanGate(d["name"], d["bool_type"], refs)
    raise ValueError(f"unknown gate type {t}")


def same_gate(a, b):
    try:
        return gate_to_dict(a, ()) == gate_to_dict(b, ())
    except TypeError:
        return False
