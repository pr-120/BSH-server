import fnmatch
import os

import matplotlib.pyplot as plt

acc_report_folder_path = "./storage"
acc_report_file_template = "accuracy-report=*=p10-acc-{}=e0.2d0.01a0.0050y0.30=Log-SiLU=h25=xavier.txt"
max_episodes = 7000

all_accuracies = []
all_episode_numbers = range(100, max_episodes+1, 100)

# for e in range(100, 100+1, 100):
for e in all_episode_numbers:
    filename = None
    for file in os.listdir(acc_report_folder_path):
        if fnmatch.fnmatch(file, acc_report_file_template.format(e)):
            filename = file
            break
    if filename is not None:
        with open(os.path.join(acc_report_folder_path, filename), "r") as report:
            content = report.read()
        sections = content.split("==========")
        trained_results_section_index = sections.index(" ACCURACY TABLE (TRAINED) ") + 1
        # print(trained_results_section_index)

        # print(sections[trained_results_section_index])
        overall_results = sections[trained_results_section_index].split("----- Overall -----")[1].strip().split("\n")
        config_3_results = overall_results[3].split(":\t")[1].split(" ")[0]
        # print(config_3_results)
        accuracy = round(float(config_3_results[:-1]) / 100, 4)
        # print(accuracy)
        all_accuracies.append(accuracy)
# print(all_accuracies)

filtered_episode_numbers = []
filtered_accuracies = []
for idx in range(0, len(all_episode_numbers)):
    if all_accuracies[idx] >= 0.99:
        filtered_episode_numbers.append(all_episode_numbers[idx])
        filtered_accuracies.append(all_accuracies[idx])

fig, (ax1, ax2) = plt.subplots(2, 1)  # 1 rows for subplots, 1 column
fig.set_size_inches(10, 10)  # width/height in inches
fig.set_tight_layout(tight=True)

ax1.scatter(all_episode_numbers, all_accuracies, s=5, color="blue")
ax1.set_ylabel("Accuracy")

ax1.xaxis.get_major_locator().set_params(integer=True)
ax1.yaxis.get_major_locator().set_params(integer=True)

ax2.scatter(filtered_episode_numbers, filtered_accuracies, s=5, color="blue")
ax2.set_ylabel("Accuracy")

ax2.set_xlabel("Episodes")

fig.align_ylabels()

fig_file = os.path.join("./storage/accuracy-fig.png")
plt.savefig(fig_file)
