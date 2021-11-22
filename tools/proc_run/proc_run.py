import subprocess


def proc_run(args):
    print(f"""[running command] {" ".join(args)}""")
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.stdout:
        print(f"[stdout returned] \n{p.stdout.decode()}")
    if p.stderr:
        print(f"[stderr returned] \n{p.stderr.decode()}")
