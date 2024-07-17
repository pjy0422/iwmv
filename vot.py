import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def iwmv_algorithm(Z, L, max_iter=100):
    M, N = Z.shape
    nu = np.ones(M)
    T = ~np.isnan(Z)

    for _ in range(max_iter):
        # Step 1: Majority voting to estimate labels
        y_hat = np.zeros(N)
        for j in range(N):
            label_counts = np.array([np.sum(nu * (Z[:, j] == k)) for k in range(L)])
            y_hat[j] = np.argmax(label_counts)

        # Step 2: Estimate worker accuracies
        w_hat = np.array([np.sum(Z[i, :] == y_hat) / np.sum(T[i, :]) for i in range(M)])
        nu = L * w_hat - 1

        # Check for convergence
        y_hat_new = np.zeros(N)
        for j in range(N):
            label_counts = np.array([np.sum(nu * (Z[:, j] == k)) for k in range(L)])
            y_hat_new[j] = np.argmax(label_counts)
        if np.array_equal(y_hat, y_hat_new):
            break

    return y_hat, w_hat


def oracle_map_rule(Z, L, worker_accuracies):
    M, N = Z.shape
    nu = (L / (L - 1)) * (L * worker_accuracies - 1)

    y_hat = np.zeros(N)
    for j in range(N):
        label_counts = np.array([np.sum(nu * (Z[:, j] == k)) for k in range(L)])
        y_hat[j] = np.argmax(label_counts)

    return y_hat


def simulate_data(M, N, L, q, avg_accuracy, tolerance=0.01, mode="a"):
    a = (2 * avg_accuracy) / (1 - avg_accuracy)

    # Ensure the average worker accuracy is within the specified tolerance
    while True:
        worker_accuracies = np.random.beta(a, 2, M)
        if np.abs(np.mean(worker_accuracies) - avg_accuracy) <= tolerance:
            break
    if mode == "b,c":
        worker_accuracies = np.random.beta(2.3, 2, M)
    Z = np.full((M, N), np.nan)  # Use np.nan to represent unlabelled items
    true_labels = np.random.randint(0, L, N)  # Generate true labels for the items

    for i in range(M):
        for j in range(N):
            if np.random.rand() < q:  # Probability q that worker i labels item j
                if np.random.rand() < worker_accuracies[i]:  # Worker labels correctly
                    Z[i, j] = true_labels[j]
                else:  # Worker labels incorrectly
                    Z[i, j] = np.random.choice(
                        [label for label in range(L) if label != true_labels[j]]
                    )

    return Z, true_labels, worker_accuracies


def compute_error_bound(L, q, nu, worker_accuracies):
    t1 = (q / ((L - 1) * np.linalg.norm(nu, 2))) * np.sum(
        nu * (L * worker_accuracies - 1)
    )
    c = np.max(nu) / np.linalg.norm(nu, 2)
    sigma2 = q

    term1 = np.exp(-(t1**2) / 2)
    term2 = np.exp(-(t1**2) / (2 * (sigma2 + c * t1 / 3)))

    return (L - 1) * min(term1, term2)


def figure_a():
    M = 31
    N = 200
    L = 3
    q = 0.3
    avg_accuracies = np.arange(0.38, 1.01, 0.05)
    num_trials = 100
    epsilon = 1e-10  # Small value to avoid log(0)

    error_rates_iwmv = []
    error_rates_orac_map = []
    error_bounds_orac_map = []
    # for (a)
    init_w_list = []
    final_w_list = []
    for avg_accuracy in tqdm(avg_accuracies):
        total_error_iwmv = 0
        total_error_orac_map = 0
        total_error_bound_orac_map = 0
        print(avg_accuracy)
        trial_init_w = np.zeros(M)
        trial_final_w = np.zeros(M)
        for _ in tqdm(range(num_trials)):
            Z, true_labels, worker_accuracies = simulate_data(
                M, N, L, q, avg_accuracy, mode="a"
            )
            trial_init_w += worker_accuracies
            # IWMV
            predicted_labels_iwmv, w_hat = iwmv_algorithm(Z, L)
            total_error_iwmv += np.mean(predicted_labels_iwmv != true_labels)
            trial_final_w += w_hat
            # Oracle MAP
            predicted_labels_orac_map = oracle_map_rule(Z, L, worker_accuracies)
            total_error_orac_map += np.mean(predicted_labels_orac_map != true_labels)

            # Oracle MAP Error Bound
            nu_orac_map = (L / (L - 1)) * (L * worker_accuracies - 1)
            error_bound_orac_map = compute_error_bound(
                L, q, nu_orac_map, worker_accuracies
            )
            total_error_bound_orac_map += error_bound_orac_map
        init_w_list.append(trial_init_w / num_trials)
        final_w_list.append(trial_final_w / num_trials)
        error_rates_iwmv.append(total_error_iwmv / num_trials)
        error_rates_orac_map.append(total_error_orac_map / num_trials)
        error_bounds_orac_map.append(total_error_bound_orac_map / num_trials)

    with open("worker_accuracies.txt", "w") as f:
        for init_w, final_w, avg_accuracy in zip(
            init_w_list, final_w_list, avg_accuracies
        ):
            f.write("Average Worker Accuracy: ")
            f.write(str(avg_accuracy))
            f.write("\n")
            f.write("Initial worker accuracies: \n")
            f.write(str(init_w))
            f.write("\n")
            f.write(f"average = {np.mean(init_w)}\n")
            f.write("Final worker accuracies: \n")
            f.write(str(final_w))
            f.write("\n\n")
    # Plotting the results
    plt.plot(
        avg_accuracies,
        np.log10(np.array(error_rates_iwmv) + epsilon),
        marker="o",
        label="IWMV",
    )
    plt.plot(
        avg_accuracies,
        np.log10(np.array(error_rates_orac_map) + epsilon),
        marker="s",
        label="Oracle MAP",
    )
    plt.plot(
        avg_accuracies,
        np.log10(np.array(error_bounds_orac_map) + epsilon),
        marker="x",
        label="Error Bound (Oracle MAP)",
        linestyle="--",
    )
    plt.xlabel("Average Worker Accuracy")
    plt.ylabel("Log10 Error Rate")
    plt.title("Error Rate vs Average Worker Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure_a.png")
    plt.show()


def figure_b():
    mode = "b,c"
    N = 200
    M_list = np.arange(10, 101, 10)
    q = 0.3
    L = 3
    num_trials = 100
    avg_accuracy = 0.5  # not used in this case
    epsilon = 1e-10  # Small value to avoid log(0)
    init_w_list = []
    final_w_list = []
    error_rates_iwmv = []
    error_rates_orac_map = []
    error_bounds_orac_map = []
    # for (b)
    for M in tqdm(M_list):
        total_error_iwmv = 0
        total_error_orac_map = 0
        total_error_bound_orac_map = 0
        for _ in tqdm(range(num_trials)):
            Z, true_labels, worker_accuracies = simulate_data(
                M, N, L, q, avg_accuracy, mode=mode
            )
            init_w_list.append(worker_accuracies)
            # IWMV
            predicted_labels_iwmv, w_hat = iwmv_algorithm(Z, L)
            total_error_iwmv += np.mean(predicted_labels_iwmv != true_labels)
            final_w_list.append(w_hat)
            # Oracle MAP
            predicted_labels_orac_map = oracle_map_rule(Z, L, worker_accuracies)
            total_error_orac_map += np.mean(predicted_labels_orac_map != true_labels)

            # Oracle MAP Error Bound
            nu_orac_map = (L / (L - 1)) * (L * worker_accuracies - 1)
            error_bound_orac_map = compute_error_bound(
                L, q, nu_orac_map, worker_accuracies
            )
            total_error_bound_orac_map += error_bound_orac_map

        error_rates_iwmv.append(total_error_iwmv / num_trials)
        error_rates_orac_map.append(total_error_orac_map / num_trials)
        error_bounds_orac_map.append(total_error_bound_orac_map / num_trials)
    for init_w, final_w in zip(init_w_list, final_w_list):
        print("Initial worker accuracies: ")
        print(init_w)
        print("Final worker accuracies: ")
        print(final_w)
    # Plotting the results
    plt.plot(
        M_list, np.log10(np.array(error_rates_iwmv) + epsilon), marker="o", label="IWMV"
    )
    plt.plot(
        M_list,
        np.log10(np.array(error_rates_orac_map) + epsilon),
        marker="s",
        label="Oracle MAP",
    )
    plt.plot(
        M_list,
        np.log10(np.array(error_bounds_orac_map) + epsilon),
        marker="x",
        label="Error Bound (Oracle MAP)",
        linestyle="--",
    )
    plt.xlabel("Number of Workers")
    plt.ylabel("Log10 Error Rate")
    plt.title("Error Rate vs Number of Workers")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure_b.png")
    plt.show()


def figure_c():
    mode = "b,c"
    N_list = np.arange(50, 1001, 50)
    M = 31
    q = 0.3
    L = 3
    num_trials = 100
    avg_accuracy = 0.5  # not used in this case
    epsilon = 1e-10  # Small value to avoid log(0)
    init_w_list = []
    final_w_list = []
    error_rates_iwmv = []
    error_rates_orac_map = []
    error_bounds_orac_map = []
    # for (b)
    for N in tqdm(N_list):
        total_error_iwmv = 0
        total_error_orac_map = 0
        total_error_bound_orac_map = 0
        for _ in tqdm(range(num_trials)):
            Z, true_labels, worker_accuracies = simulate_data(
                M, N, L, q, avg_accuracy, mode=mode
            )
            init_w_list.append(worker_accuracies)
            # IWMV
            predicted_labels_iwmv, w_hat = iwmv_algorithm(Z, L)
            total_error_iwmv += np.mean(predicted_labels_iwmv != true_labels)
            final_w_list.append(w_hat)
            # Oracle MAP
            predicted_labels_orac_map = oracle_map_rule(Z, L, worker_accuracies)
            total_error_orac_map += np.mean(predicted_labels_orac_map != true_labels)

            # Oracle MAP Error Bound
            nu_orac_map = (L / (L - 1)) * (L * worker_accuracies - 1)
            error_bound_orac_map = compute_error_bound(
                L, q, nu_orac_map, worker_accuracies
            )
            total_error_bound_orac_map += error_bound_orac_map

        error_rates_iwmv.append(total_error_iwmv / num_trials)
        error_rates_orac_map.append(total_error_orac_map / num_trials)
        error_bounds_orac_map.append(total_error_bound_orac_map / num_trials)
    for init_w, final_w in zip(init_w_list, final_w_list):
        print("Initial worker accuracies: ")
        print(init_w)
        print("Final worker accuracies: ")
        print(final_w)
    # Plotting the results
    plt.plot(
        N_list, np.log10(np.array(error_rates_iwmv) + epsilon), marker="o", label="IWMV"
    )
    plt.plot(
        N_list,
        np.log10(np.array(error_rates_orac_map) + epsilon),
        marker="s",
        label="Oracle MAP",
    )
    plt.plot(
        N_list,
        np.log10(np.array(error_bounds_orac_map) + epsilon),
        marker="x",
        label="Error Bound (Oracle MAP)",
        linestyle="--",
    )
    plt.xlabel("Number of Items")
    plt.ylabel("Log10 Error Rate")
    plt.title("Error Rate vs Number of Items")
    plt.legend()
    plt.grid(True)
    plt.savefig("figure_c.png")
    plt.show()


def merge_figures(figures, output_path):
    images = [Image.open(figure) for figure in figures]
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    new_img.save(output_path)


if __name__ == "__main__":
    figure_a()
    """figure_b()
    figure_c()

    figures = ["figure_a.png", "figure_b.png", "figure_c.png"]
    merge_figures(figures, "merged_figure.png")"""
