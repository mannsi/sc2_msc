import os
import csv


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
