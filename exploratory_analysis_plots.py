import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import pandas as pd


class Plots:
    def histogram_DrugUse(dataset, categories):
        sns.set(style="darkgrid")

        variables = [0, 0, 0, 0, 0, 0]
        variables_overclaiming = [0, 0, 0, 0, 0, 0]

        num_categories = len(categories)

        bar_positions = np.arange(num_categories)

        for row in dataset:
            for i in range(8, 14):
                # daily users who do not claim to have used the fictional drug
                if row[i] == "1" and row[14] != "1":
                    variables[i - 8] += 1
                    variables_overclaiming[i - 8] += 1

                # users who claims to have used the fictional drug
                elif row[i] == "1" and row[14] == "1":
                    variables_overclaiming[i - 8] += 1

        print(variables)
        print(variables_overclaiming)

        plt.bar(
            bar_positions,
            variables_overclaiming,
            alpha=0.3,
        )

        plt.bar(
            bar_positions,
            variables,
            alpha=1.0,
        )

        plt.xticks(ticks=range(len(categories)), labels=categories)

        plt.xlabel("Substances")
        plt.ylabel("Users in the last day")
        plt.title("People that have used a specific drug during the last day")

        plt.show()

    def piechart_extraversion(dataset):
        extraversion_values = []
        values = [0, 0, 0]
        for row in dataset:
            extraversion_values.append(float(row[2]))

            for e in extraversion_values:
                if e < 0.00016 - 0.99745:
                    values[0] += 1
                elif e <= 0.00016 + 0.99745 and e >= 0.00016 - 0.99745:
                    values[1] += 1
                elif e > 0.00016 + 0.99745:
                    values[2] += 1

        class_counts = Counter(extraversion_values)

        class_labels = ["low", "medium,", "high"]
        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=class_labels, startangle=140)
        plt.title("Pie Chart for Extraversion scores among all the instances")
        plt.axis("equal")

    def piechart_extr_subst(dataset, categories):
        j = 0
        for i in range(8, 14):
            extraversion_values = []
            values = [0, 0, 0]
            for row in dataset:
                if row[i] == "1" and row[14] == "0":
                    extraversion_values.append(float(row[2]))

            for e in extraversion_values:
                if e < 0.00016 - 0.99745:
                    values[0] += 1
                elif e <= 0.00016 + 0.99745 and e >= 0.00016 - 0.99745:
                    values[1] += 1
                elif e > 0.00016 + 0.99745:
                    values[2] += 1

            class_counts = Counter(extraversion_values)

            class_labels = ["low", "medium,", "high"]
            plt.figure(figsize=(6, 6))
            plt.pie(values, labels=class_labels, startangle=140)
            plt.title("Pie Chart for Extraversion Scores " + categories[j])
            plt.axis("equal")
            j += 1

    def histograms_personality_density(dataset, scores):
        fig, axes = plt.subplots(2, 4, figsize=(15, 10))

        for idx, ((score, (mean, std_dev, min_val, max_val)), ax) in enumerate(
            zip(scores.items(), axes.flatten())
        ):
            samples = [float(row[idx + 1]) for row in dataset]

            num_unique_elem = len(set(samples))

            # set number of bars and bar width

            bar_width = 3 / num_unique_elem
            num_bins = int((max_val - min_val) / bar_width) + 1
            bins = [min_val + i * bar_width for i in range(num_bins + 1)]

            # Filter values within one standard deviation
            in_range = [
                val for val in samples if mean - std_dev <= val <= mean + std_dev
            ]

            # Filter values outside one standard deviation
            out_range = [
                val for val in samples if val < mean - std_dev or val > mean + std_dev
            ]
            colors = sns.color_palette("deep")
            # Plot bars within one standard deviation in one color
            ax.hist(in_range, bins=bins, alpha=1.0, color=colors[0])

            # Plot bars outside one standard deviation in another color
            ax.hist(out_range, bins=bins, alpha=1.0, color=colors[1])

            ax.set_title(score)
            ax.set_xlabel("Value")
            ax.set_ylabel("Nr_istances")
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def heatmap(dataset, scores):
        df = pd.read_csv(dataset, delimiter=",", header=None)

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.drop(df.columns[[0, 8, 9, 10, 11, 12, 13, 14]], axis=1)
        #  Calculate the correlation matrix
        correlation_matrix = df.corr()

        # Visualize the correlation matrix
        plt.figure(figsize=(10, 8))
        var_names = scores.keys()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            alpha=0.6,
            xticklabels=var_names,
            yticklabels=var_names,
        )

        plt.title("Correlation Matrix")
        plt.show()
