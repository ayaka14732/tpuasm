"""以单字段穷举、非零组合和随机操作数重新核对 encoder 拒绝的槽内形式。"""
import argparse
from importlib.metadata import distribution, version
import json
from pathlib import Path
import random
from typing import TypedDict
from generate_tpu_v6e_tc_isa import SPEC
from ghostlite_isa import Isa, bundle, descriptors, encode
from tpuasm.tpu_v6e_tc_isa_data import REJECTED

class Control(TypedDict):
    slot: str
    accepted: bool

class Rejection(TypedDict):
    slot: str
    form: str
    cases: int
    accepted: list[int]
    controls: list[Control]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/tpuasm-v6e-rejections.json'))
    args = parser.parse_args()
    isa = Isa(SPEC, *descriptors(Path(str(distribution('libtpu').locate_file('libtpu/libtpu.so')))))
    shared = dict(zip((name for name, _, _ in SPEC.shared), (0xa1357, 0x5b246, 0xc369a, 0x7d48b, 0xe5abc, 0x9f6de, 20, 21, 22, 23)))
    rng = random.Random(9817)
    results: list[Rejection] = []
    for slot, names in REJECTED.items():
        for name in names:
            form = next(f for f in isa.forms(slot) if f.name == name)
            operands = isa.operands(form)
            cases: list[dict[str, int]] = [{}]
            domains = {}
            for field in operands:
                values = isa.enum_values(field) if field.kind == 14 else [0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 65535, 1048575, 0xffffffff]
                domains[field.name] = values
                cases.extend({field.name: value} for value in values)
            cases.extend({field.name: value for field in operands if field.kind != 14} for value in (1, 3, 7, 15, 31))
            cases.extend({field.name: rng.choice(domains[field.name]) for field in operands} for _ in range(128))
            words = encode(SPEC, [bundle(SPEC, isa.slot_bundle(slot, form, values), shared) for values in cases])
            accepted = [i for i, word in enumerate(words) if word is not None]
            controls: list[Control] = []
            for other in isa.slot_messages:
                if other == slot or name in REJECTED.get(other, ()):
                    continue
                alternative = next((f for f in isa.forms(other) if f.name == name), None)
                if alternative is not None:
                    control = encode(SPEC, [bundle(SPEC, isa.slot_bundle(other, alternative, {field.name: 1 for field in operands if field.kind != 14}))])[0]
                    controls.append({'slot': other, 'accepted': control is not None})
            result: Rejection = {'slot': slot, 'form': name, 'cases': len(cases), 'accepted': accepted, 'controls': controls}
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps({'libtpu': version('libtpu'), 'seed': 9817, 'results': results}, indent=2) + '\n')
    assert all(not result['accepted'] and result['controls'] and all(control['accepted'] for control in result['controls']) for result in results)
    print(f'{len(results)} rejected slot forms, {sum(result["cases"] for result in results)} operand cases; all positive controls accepted')

if __name__ == '__main__':
    main()
