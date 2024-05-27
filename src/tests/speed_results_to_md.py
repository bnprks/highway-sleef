import argparse
import collections
import json
import math
import re

def main():
    parser = argparse.ArgumentParser(description="Convert results from measure_speed --benchmark_format=json to tsv table")
    parser.add_argument("input_json", help="Path of json from measure_speed --benchmark_format=json")
    args = parser.parse_args()

    data = json.load(open(args.input_json))

    NUMS_PER_ITERATION = 1 <<12

    timing_data = collections.defaultdict(lambda: collections.defaultdict(lambda: math.nan))

    name_pattern = re.compile(r"BM_op_(?P<type>float|double)/(?P<op>[A-Z][A-EG-Za-z0-9]+)(?P<tool>(Fast)?(Sleef|Translated|Hwy|Std))")

    for b in data["benchmarks"]:
        assert b["time_unit"] == "ns"

        name_info = name_pattern.match(b["name"]).groupdict()
        op_name = name_info["op"] + "_" + name_info["type"] 
        run_version = 1
        while name_info["tool"] in timing_data[op_name]:
            run_version += 1
            op_name = name_info["op"] + "_" + name_info["type"] + "_v" + str(run_version)
        
        timing_data[op_name][name_info["tool"]] = b["cpu_time"]

    print("Op\tType\tRange tested\tScalar\tHwy\tTranslated 1 ULP\tSleef 1 ULP\tTranslated 3.5 ULP\tSleef 3.5 ULP")

    for op, d in timing_data.items():
        if "_float" in op:
            op_type = "f32"
            op = op.replace("_float", "")
        elif "_double" in op:
            op_type = "f64"
            op = op.replace("_double", "")
        else:
            assert False
        raw_time = [d["Std"], d['Hwy'], d['Translated'], d['Sleef'], d['FastTranslated'], d['FastSleef']]
        per_elem = [x/NUMS_PER_ITERATION for x in raw_time]
        reference = min(x for x in raw_time if not math.isnan(x))
        relative = [x / reference for x in raw_time]

        entries = "\t".join([f"{x:.1f} ns ({y:.1f}x)" if not math.isnan(x) else "N/A" for x, y in zip(per_elem, relative)])
        print(f"{op}\t{op_type}\t\t{entries}")


    

if __name__ == "__main__":
    main()