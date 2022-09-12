from subprocess import Popen, PIPE, STDOUT, run
import subprocess


def start(params, time_limit, filename, solver):

    if solver == "cadical":

        proc = subprocess.Popen(
            [
                "solvers/./cadical",
                "--no-witness",
                *params,
                "-t",
                str(time_limit),
                f"{filename}",
            ],
            stdout=PIPE,
        )
        return proc

    elif solver == "glucose":

        proc = subprocess.Popen(
            [
                "solvers/./glucose_static",
                "-cpu-lim=" + str(time_limit),
                *params,
                f"{filename}",
            ],
            stdout=PIPE,
        )
        return proc

    elif solver == "cplex":

        dir_file = list(filename.split("/"))
        directory = dir_file[0]
        filename = dir_file[1]

        proc = subprocess.Popen(
            [
                "python3",
                "solvers/cplex_solve.py",
                "-d",
                directory,
                "-f",
                f"{filename}",
                "-to",
                str(time_limit),
                "-g",
                str(params),
            ],
            stdout=PIPE,
        )

        return proc


def check_output(line, interim, solver):

    if solver == "cadical":

        if line != b"":
            b = str(line.strip())
            # Check for progress
            if b[2] == "c":
                if len(b) > 4:
                    b = list(b.split(" "))
                    b = list(filter(lambda a: a != "", b))
                    if len(b) > 13:
                        interim = [
                            int(b[4]),
                            int(b[5]),
                            int(b[6]),
                            int(b[7]),
                            int(b[11]),
                            int(b[13][:-2]),
                        ]

                        return interim
        else:
            return "No output"

    elif solver == "glucose":

        if line != b"":
            b = str(line.strip())
            if b[2] == "c":
                b = list(b.split("  "))
                b = list(filter(lambda a: a != "", b))
                if len(b) > 11 and len(b[11]) > 5:
                    interim = [
                        int(b[2][1:]),
                        int(b[3][:-2]),
                        int(b[4][1:]),
                        int(b[5][1:]),
                        int(b[6][1:-2]),
                        int(b[7]),
                        int(b[8]),
                        int(b[9]),
                        int(b[10][:-1]),
                        float(b[11][:-4]),
                    ]
                    return interim
        else:
            return "No output"

    elif solver == "cplex":

        if line != b"":
            currentobj = interim[0]
            gap = interim[1]
            bestbound = interim[2]
            numbernodes = interim[3]
            b = str(line.strip())
            if b[2:4] == "CI":
                b = b[6:-1]
                currentobj = float(b)
            elif b[2] == "G":
                a = float(b[6:-1])
                gap = int(a)
            elif b[2] == "B":
                b = b[2:-1]
                bestbound = float(b[4:])
            elif b[2:4] == "CN":
                a = float(b[7:-1])
                numbernodes = a
            interim = [currentobj, gap, bestbound, numbernodes]

            return interim
        else:
            return "No output"


def check_if_solved(line, results, proc, event, non_nlock_read, solver):

    if solver == "cadical":

        if line != b"":
            b = str(line.strip())
            if b[2] == "s":
                results[0] = 1
                time_not_given = True
                while time_not_given:
                    line = non_nlock_read(proc.stdout)
                    b = str(line.strip())
                    if len(b) > 4:
                        if b[4:17] == "total process":
                            b = list(b.split("  "))
                            b = list(filter(lambda a: a != "", b))
                            results[1] = float(b[1])
                            time_not_given = False

                event[0] = 1
                proc.stdout.close()

            return results, event
        else:
            return "No output"

    elif solver == "glucose":

        if line != b"":
            b = str(line.strip())
            if len(b) > 4:
                if b[4] == "C":
                    b = list(b.split("  "))
                    b = b[7].split()
                    results[1] = float(b[1])
                    result_not_given = True
                    while result_not_given:
                        line = non_nlock_read(proc.stdout)
                        b = str(line.strip())
                        if len(b) > 4:
                            if b[2] == "s" and b[4] != "I":
                                results[0] = 1
                                result_not_given = 0
                                event[0] = 1
                                proc.stdout.close()

            return results, event
        else:
            return "No output"

    elif solver == "cplex":

        if line != b"":
            b = str(line.strip())
            if b[2] == "[":
                b = b[2:-3]
                b = list(b.split(", "))
                results[:] = [float(b[0][1:]), float(b[1][:-1])]
                event[0] = 1
                proc.stdout.close()

            return results, event
        else:
            return "No output"
