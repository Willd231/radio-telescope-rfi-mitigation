#!/home/wdellinger/anaconda3/bin/python
import sys
import subprocess


usage = "Usage: python CudaRunner.py 'both/run' 'old/testing/block/dual' 'A/B/C' {file name}"

if len(sys.argv) < 5:
    print(usage)
    sys.exit(1)


runType = sys.argv[1]
mode = sys.argv[2]
prod_type = sys.argv[3]
file = sys.argv[4]

nchan = 8

if prod_type not in ["A", "B", "C"]:
    print(usage)
    sys.exit(1)

if mode not in ["old", "testing", "block", "dual"]:
    print(usage)
    sys.exit(1)


print("You selected mode: ", mode)

match(mode):
    case "old":
        compile_cmd = "nvcc -O3 corr_gpu_complex_v2.cu -lcufft -o corr_gpu_v2"
        run_cmd = "./corr_gpu_v2 -p " + prod_type + " -c 1024 -n 4 -i /mnt/carsedat/20250220_181537_0000.dat -o output -d"
    case "testing":
        compile_cmd = "nvcc -O3 test_gpu.cu -lcufft -o test_gpu"
        run_cmd = "./test_gpu -p " + prod_type + " -c 1024 -n 4 -i /mnt/carsedat/20250220_181537_0000.dat -o output -d"
    case "block":
        compile_cmd = "nvcc -O3 block_gpu_complex.cu -lcufft -o block_gpu"
        run_cmd = "./block_gpu -p " + prod_type + " -c 1024 -n 4 -i /mnt/carsedat/20250917_174710_0000.datnp0 -o output_6 -d"
    case "dual":
        compile_cmd = "nvcc -O3 dualfpga_complex.cu -lcufft -o dual_gpu"
        run_cmd = f"./dual_gpu -p {prod_type} -c 1024 -n {nchan} -i /mnt/carsedat/{file} -o output_6 -d"
    case _:
        print("Invalid mode. Exiting...")
        sys.exit(1)
    

if runType == "both":
    # Compile
    subprocess.run(compile_cmd, shell=True)
    # Run
    subprocess.run(run_cmd, shell=True)
elif runType == "run":
    subprocess.run(run_cmd, shell=True)
else:
    print("Invalid run type. Exiting...")
    sys.exit(1)
