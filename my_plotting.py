import os
import csv

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker


def save_results_to_file(file_name, move_steps_after_dmg, num_move_steps, steps_until_dmg):
    file_path = os.path.join('results',file_name)
    if not os.path.exists(file_path):
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ['move_steps_after_dmg', 'num_move_steps', 'steps_until_dmg'])

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([move_steps_after_dmg, num_move_steps, steps_until_dmg])
        csvfile.flush()


def plot_ssm_results(file_name, output_file_name, plt_title):
    columns = []

    with open(os.path.join('results',file_name), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for column in zip(*[line for line in reader]):
            columns.append(list(column))

    move_steps_after_dmg = columns.pop(0)
    move_steps_after_dmg.pop(0)  # Remove header
    move_steps_after_dmg = [float(x) for x in move_steps_after_dmg]

    num_move_steps = columns.pop(0)
    num_move_steps.pop(0)  # Remove header

    steps_until_dmg = columns.pop(0)
    steps_until_dmg.pop(0)  # Remove header
    steps_until_dmg = [float(x) for x in steps_until_dmg]

    fig = plt.figure()
    plt.xlabel('Wait steps after dmg')
    plt.ylabel('Num steps until dmg')
    plt.plot(move_steps_after_dmg, steps_until_dmg, 'o')

    plt.gcf().set_size_inches(10, 15)
    plt.title(plt_title)

    plt.show()
    # plt.savefig(output_file_name)


if __name__ == "__main__":
    plot_ssm_results('ssm_move_steps.txt', 'ssm_move_steps.png', 'ssm_move_steps')
