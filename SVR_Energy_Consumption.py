{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPwptRthg86tZ/vXr901Zw/",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/giacovides/FYP/blob/main/SVR_Energy_Consumption.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OXvjVC__kiY9",
        "outputId": "fa5e17c3-c653-47fa-9e2a-aaaeedd8ab86"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "# Define the file paths\n",
        "file_path1 = '/content/drive/MyDrive/SuperUROP /Data Analysis/caltech_training_data.csv'\n",
        "file_path2 = '/content/drive/MyDrive/SuperUROP /Data Analysis/caltech_testing_data.csv'\n",
        "file_path3 = '/content/drive/MyDrive/SuperUROP /Data Analysis/JPL_training_data.csv'\n",
        "file_path4  = '/content/drive/MyDrive/SuperUROP /Data Analysis/JPL_testing_data.csv'\n",
        "# Use pandas to read the CSV files and then convert them to NumPy arrays\n",
        "caltech_train = pd.read_csv(file_path1).values\n",
        "caltech_test = pd.read_csv(file_path2).values\n",
        "\n",
        "JPL_train = pd.read_csv(file_path3).values\n",
        "JPL_test=pd.read_csv(file_path4).values"
      ],
      "metadata": {
        "id": "YnQYa1tRkozO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Data Processing for SVR"
      ],
      "metadata": {
        "id": "Lk3hjf9snXAi"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Remove row number (in 1st column)\n",
        "caltech_train=caltech_train[:,1:]\n",
        "caltech_test=caltech_test[:,1:]\n",
        "\n",
        "JPL_train=JPL_train[:,1:]\n",
        "JPL_test=JPL_test[:,1:]"
      ],
      "metadata": {
        "id": "Dbewg3sKnUxD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Remove departure time (2nd column)\n",
        "# Convert arrival date to hour and find day of the week\n",
        "from datetime import datetime\n",
        "\n",
        "def convert_time_and_day(data_array):\n",
        "    \"\"\"\n",
        "    Converts the time from HH:MM to HH.XX format and appends the day of the week to it.\n",
        "    Also, removes the second column.\n",
        "    \"\"\"\n",
        "    transformed_data = []\n",
        "    for row in data_array:\n",
        "        # Convert the arrival time to HH.XX format\n",
        "        time_obj = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')\n",
        "        new_time = time_obj.hour + (time_obj.minute / 60.0)\n",
        "\n",
        "        # Convert the date to a day of the week\n",
        "        day_of_week = time_obj.strftime('%A')\n",
        "        new_time = str(new_time) + \" \" + day_of_week\n",
        "\n",
        "        # Create a new row excluding the second column\n",
        "        new_row = [new_time] + list(row[2:])\n",
        "        transformed_data.append(new_row)\n",
        "\n",
        "    return np.array(transformed_data)\n",
        "\n",
        "caltech_train=convert_time_and_day(caltech_train)\n",
        "caltech_test=convert_time_and_day(caltech_test)\n",
        "JPL_train=convert_time_and_day(JPL_train)\n",
        "JPL_test=convert_time_and_day(JPL_test)"
      ],
      "metadata": {
        "id": "CghHjHlxndsP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def day_to_number(day):\n",
        "    \"\"\"Converts a day of the week to its corresponding discrete value.\"\"\"\n",
        "    days = {\n",
        "        'Monday': 1,\n",
        "        'Tuesday': 2,\n",
        "        'Wednesday': 3,\n",
        "        'Thursday': 4,\n",
        "        'Friday': 5,\n",
        "        'Saturday': 6,\n",
        "        'Sunday': 7\n",
        "    }\n",
        "    return days[day]\n",
        "\n",
        "def separate_time_and_day(data_array):\n",
        "    \"\"\"\n",
        "    Separates the time and day in the given column,\n",
        "    and converts the day into a discrete value between 1 and 7.\n",
        "\n",
        "    \"\"\"\n",
        "    transformed_data = []\n",
        "    for row in data_array:\n",
        "        time_day_str = row[0]\n",
        "        time, day = time_day_str.split()\n",
        "        time = float(time)\n",
        "        day_num = day_to_number(day)\n",
        "\n",
        "        # Create a new row with separated time and day number\n",
        "        new_row = [time, day_num] + list(row[1:])\n",
        "        transformed_data.append(new_row)\n",
        "\n",
        "    return np.array(transformed_data)\n",
        "\n",
        "caltech_train=separate_time_and_day(caltech_train)\n",
        "caltech_test=separate_time_and_day(caltech_test)\n",
        "JPL_train=separate_time_and_day(JPL_train)\n",
        "JPL_test=separate_time_and_day(JPL_test)"
      ],
      "metadata": {
        "id": "jDMe8JaEnpEs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Make training and testing set have the same user IDs\n",
        "users_from_training_caltech = set(caltech_train[:, 3])\n",
        "mask_caltech = np.isin(caltech_test[:, 3], list(users_from_training_caltech))\n",
        "caltech_test = caltech_test[mask_caltech]\n",
        "users_from_testing_caltech = set(caltech_test[:, 3])\n",
        "mask_caltech = np.isin(caltech_train[:, 3], list(users_from_testing_caltech))\n",
        "caltech_train = caltech_train[mask_caltech]\n",
        "\n",
        "users_from_training = set(JPL_train[:, 3])\n",
        "mask = np.isin(JPL_test[:, 3], list(users_from_training))\n",
        "JPL_test = JPL_test[mask]\n",
        "users_from_testing = set(JPL_test[:, 3])\n",
        "mask = np.isin(JPL_train[:, 3], list(users_from_testing))\n",
        "JPL_train = JPL_train[mask]"
      ],
      "metadata": {
        "id": "Qa1xVIY4nq_C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "caltech_train = np.array(caltech_train, dtype='float')\n",
        "caltech_test = np.array(caltech_test, dtype='float')\n",
        "JPL_train = np.array(JPL_train, dtype='float')\n",
        "JPL_test = np.array(JPL_test, dtype='float')"
      ],
      "metadata": {
        "id": "PWHdbrPsnsxG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Predict energy consumption using K-fold CV & Grid Search"
      ],
      "metadata": {
        "id": "D6ep4m90-7zT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVR\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "def optimize_svr_rbf(train_data, test_data, user_id,k=5):\n",
        "\n",
        "    # Filter training and testing data for the specific user\n",
        "    user_train_data = train_data[train_data[:, 3] == user_id]\n",
        "    user_test_data = test_data[test_data[:, 3] == user_id]\n",
        "\n",
        "    # Extract columns for stay duration and energy consumption for both train and test sets\n",
        "    X_train = user_train_data[:,4].reshape(-1, 1)  # Stay Duration\n",
        "    y_train = user_train_data[:,2]  # Energy Consumption\n",
        "\n",
        "    X_test = user_test_data[:,4].reshape(-1, 1)\n",
        "    y_test = user_test_data[:,2]\n",
        "\n",
        "    # Define the hyperparameters to be optimized\n",
        "    param_grid = {\n",
        "        'C': [0.1, 1, 10, 100],\n",
        "        'epsilon': [0.001, 0.01, 0.1, 1],\n",
        "        'gamma': ['scale', 'auto', 0.1, 1, 10]\n",
        "    }\n",
        "\n",
        "    # Initialize SVR with RBF kernel\n",
        "    svr_rbf = SVR(kernel='rbf')\n",
        "\n",
        "    # Initialize GridSearchCV with k-fold cross-validation\n",
        "    grid_search = GridSearchCV(estimator=svr_rbf, param_grid=param_grid, cv=k, scoring='neg_mean_squared_error', n_jobs=-1)\n",
        "\n",
        "    # Fit the model to the training data\n",
        "    grid_search.fit(X_train, y_train)\n",
        "\n",
        "    # Use the best estimator to predict on the test data\n",
        "    test_predictions = grid_search.best_estimator_.predict(X_test)\n",
        "\n",
        "    # Calculate MSE on test data\n",
        "    mse = mean_squared_error(y_test, test_predictions)\n",
        "\n",
        "    # Calculate user SMAPE\n",
        "    n = len(y_test)\n",
        "    smape_val = (1/ n) * np.sum(np.abs(y_test - test_predictions) / (np.abs(y_test+test_predictions)))*100\n",
        "\n",
        "    return grid_search.best_params_, smape_val\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "JCpYvHX7VYzn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Test the function\n",
        "user_ids_caltech = np.unique(np.concatenate((caltech_train[:, 3], caltech_test[:, 3])))\n",
        "user_ids_JPL = np.unique(np.concatenate((JPL_train[:, 3], JPL_test[:, 3])))\n",
        "\n",
        "smape_list_caltech=[]\n",
        "smape_list_JPL=[]\n",
        "best_params_caltech=[]\n",
        "best_params_JPL=[]\n",
        "for user_id in user_ids_caltech:\n",
        "    best_params,smape = optimize_svr_rbf(caltech_train, caltech_test, user_id)\n",
        "    smape_list_caltech.append(smape)\n",
        "    best_params_caltech.append(best_params)\n",
        "for user_id in user_ids_JPL:\n",
        "    best_params, smape = optimize_svr_rbf(JPL_train, JPL_test, user_id)\n",
        "    smape_list_JPL.append(smape)\n",
        "    best_params_JPL.append(best_params)"
      ],
      "metadata": {
        "id": "P5GUUUyMAE0K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Calculate average SMAPE for each location\n",
        "no_caltech_users=len(user_ids_caltech)\n",
        "caltech_smape=sum(smape_list_caltech)/no_caltech_users\n",
        "\n",
        "no_JPL_users=len(user_ids_JPL)\n",
        "JPL_smape=sum(smape_list_JPL)/no_JPL_users\n",
        "\n",
        "print(f\"Caltech SMAPE (K-fold CV): {caltech_smape}\")\n",
        "print(f\"JPL SMAPE (K-fold CV): {JPL_smape}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OD_FTvCXAEBZ",
        "outputId": "207396a1-c21e-4a89-ed3a-e3a778ccc617"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Caltech SMAPE (K-fold CV): 18.2325808923074\n",
            "JPL SMAPE (K-fold CV): 12.143215252308854\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Predict energy consumption using Leave One Out CV & Grid Search"
      ],
      "metadata": {
        "id": "9TKDd3xn7aJS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVR\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.model_selection import LeaveOneOut\n",
        "\n",
        "def optimize_svr_rbf(train_data, test_data, user_id, k=5):\n",
        "\n",
        "    # Filter training and testing data for the specific user\n",
        "    user_train_data = train_data[train_data[:, 3] == user_id]\n",
        "    user_test_data = test_data[test_data[:, 3] == user_id]\n",
        "\n",
        "    # Extract columns for stay duration and energy consumption for both train and test sets\n",
        "    X_train = user_train_data[:,4].reshape(-1, 1)  # Stay Duration\n",
        "    y_train = user_train_data[:,2]  # Energy Consumption\n",
        "\n",
        "    X_test = user_test_data[:,4].reshape(-1, 1)\n",
        "    y_test = user_test_data[:,2]\n",
        "\n",
        "    # Define the hyperparameters to be optimized\n",
        "    param_grid = {\n",
        "        'C': [0.1, 1, 10, 100],\n",
        "        'epsilon': [0.001,0.01, 0.1, 1],\n",
        "        'gamma': ['scale', 'auto', 0.1, 1, 10]\n",
        "    }\n",
        "\n",
        "    # Initialize SVR with RBF kernel\n",
        "    svr_rbf = SVR(kernel='rbf')\n",
        "\n",
        "    # Use LeaveOneOut for cross-validation\n",
        "    loo = LeaveOneOut()\n",
        "\n",
        "    # Initialize GridSearchCV with k-fold cross-validation\n",
        "    grid_search = GridSearchCV(estimator=svr_rbf, param_grid=param_grid, cv=loo, scoring='neg_mean_squared_error', n_jobs=-1)\n",
        "\n",
        "    # Fit the model to the training data\n",
        "    grid_search.fit(X_train, y_train)\n",
        "\n",
        "    # Use the best estimator to predict on the test data\n",
        "    test_predictions = grid_search.best_estimator_.predict(X_test)\n",
        "\n",
        "    # Calculate MSE on test data\n",
        "    mse = mean_squared_error(y_test, test_predictions)\n",
        "\n",
        "    # Calculate user SMAPE\n",
        "    n = len(y_test)\n",
        "    smape_val = (1/ n) * np.sum(np.abs(y_test - test_predictions) / (np.abs(y_test+test_predictions)))*100\n",
        "\n",
        "    return grid_search.best_params_, smape_val\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "8CUq6CNzW-fI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Test the function\n",
        "user_ids_caltech = np.unique(np.concatenate((caltech_train[:, 3], caltech_test[:, 3])))\n",
        "user_ids_JPL = np.unique(np.concatenate((JPL_train[:, 3], JPL_test[:, 3])))\n",
        "\n",
        "smape_list_caltech=[]\n",
        "smape_list_JPL=[]\n",
        "best_params_caltech=[]\n",
        "best_params_JPL=[]\n",
        "for user_id in user_ids_caltech:\n",
        "    best_params,smape = optimize_svr_rbf(caltech_train, caltech_test, user_id)\n",
        "    smape_list_caltech.append(smape)\n",
        "    best_params_caltech.append(best_params)\n",
        "for user_id in user_ids_JPL:\n",
        "    best_params, smape = optimize_svr_rbf(JPL_train, JPL_test, user_id)\n",
        "    smape_list_JPL.append(smape)\n",
        "    best_params_JPL.append(best_params)"
      ],
      "metadata": {
        "id": "XIhib2HXqKhk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Calculate average SMAPE for each location\n",
        "no_caltech_users=len(user_ids_caltech)\n",
        "caltech_smape=sum(smape_list_caltech)/no_caltech_users\n",
        "\n",
        "no_JPL_users=len(user_ids_JPL)\n",
        "JPL_smape=sum(smape_list_JPL)/no_JPL_users\n",
        "\n",
        "print(f\"Caltech SMAPE (Leave-One-Out CV): {caltech_smape}\")\n",
        "print(f\"JPL SMAPE (Leave-One-Out CV): {JPL_smape}\")"
      ],
      "metadata": {
        "id": "RnYJU3pkvNcI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Plot SMAPE vs R_de for all users (JPL)"
      ],
      "metadata": {
        "id": "bY3vaavuJD-Z"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize dictionaries to store user-wise SMAPE values\n",
        "user_smape_JPL = {}\n",
        "user_ids_JPL = np.unique(np.concatenate((JPL_train[:, 3], JPL_test[:, 3])))\n",
        "\n",
        "# Loop through each user ID in the JPL dataset\n",
        "for user_id in user_ids_JPL:\n",
        "    best_params, smape = optimize_svr_rbf(JPL_train, JPL_test, user_id)\n",
        "    user_smape_JPL[user_id] = smape  # Store the SMAPE in the dictionary\n",
        "\n",
        "average_smape_JPL = np.mean(list(user_smape_JPL.values()))\n",
        "\n",
        "print(f\"Average SMAPE for JPL dataset: {average_smape_JPL:.2f}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4bVysDV8JJZh",
        "outputId": "b9d6fdbc-1260-40ca-bc1e-f0d3eec52e55"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Average SMAPE for JPL dataset: 12.14%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "R_DE={'169': 2.119068507176728,\n",
        " '171': 3.7115405017002776,\n",
        " '176': 2.9977726188370353,\n",
        " '220': 4.170165871125364,\n",
        " '322': 2.974319275358279,\n",
        " '334': 1.54796573662954,\n",
        " '335': 3.9915670302142447,\n",
        " '346': 4.687498008228514,\n",
        " '365': 4.400963644915041,\n",
        " '368': 3.4312414172978474,\n",
        " '372': 2.9575939245001077,\n",
        " '374': 3.839538311235034,\n",
        " '378': 3.1515844041039993,\n",
        " '382': 4.148262279562261,\n",
        " '404': 3.097855936467447,\n",
        " '405': 4.732202935285684,\n",
        " '406': 1.158211936837657,\n",
        " '409': 2.886901319079918,\n",
        " '410': 4.280915725084575,\n",
        " '416': 4.148262279562261,\n",
        " '436': 3.3808089672729613,\n",
        " '444': 3.0622185077071102,\n",
        " '458': 4.148262279562261,\n",
        " '467': 3.777008295296614,\n",
        " '474': 2.6165283654578775,\n",
        " '476': 3.7356597962280196,\n",
        " '481': 4.045488808054726,\n",
        " '483': 2.755038189921663,\n",
        " '507': 3.7804325624313817,\n",
        " '526': 2.4478241428939165,\n",
        " '531': 3.5032226226713927,\n",
        " '537': 3.1955976636365735,\n",
        " '551': 4.767567980735159,\n",
        " '553': 2.555583224030839,\n",
        " '576': 3.380624050295972,\n",
        " '577': 4.031441936975631,\n",
        " '581': 2.9304059447845647,\n",
        " '592': 3.056683078202939,\n",
        " '607': 4.625780698041485,\n",
        " '651': 3.492080210083478,\n",
        " '726': 3.2841952906487855,\n",
        " '742': 3.9023067062499264,\n",
        " '826': 3.375181253576597,\n",
        " '933': 3.9394875755880188}"
      ],
      "metadata": {
        "id": "iTR6CWyqJKcm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Convert the keys of R_SD to integers for proper comparison\n",
        "R_DE_int_keys = {int(k): v for k, v in R_DE.items()}\n",
        "\n",
        "# Finding common keys between the two dictionaries\n",
        "common_keys = set(user_smape_JPL.keys()).intersection(R_DE_int_keys.keys())\n",
        "x_values = [R_DE_int_keys[key] for key in common_keys]\n",
        "y_values = [user_smape_JPL[key] for key in common_keys]\n",
        "\n",
        "correlation_coefficient = np.corrcoef(x_values, y_values)[0, 1]\n",
        "print(correlation_coefficient)\n",
        "\n",
        "# Calculating the line of best fit\n",
        "m, b = np.polyfit(x_values, y_values, 1)  # m is slope, b is y-intercept\n",
        "# Generating y-values for the line of best fit\n",
        "fit_line = [m*x + b for x in x_values]\n",
        "\n",
        "# Plotting\n",
        "plt.figure(figsize=(10, 6))\n",
        "plt.scatter(x_values, y_values, color='blue')\n",
        "plt.plot(x_values, fit_line, color='black')\n",
        "plt.title(\"Comparison of JPL SMAPE and R_DE values for each user\")\n",
        "plt.xlabel(\"R_SD\")\n",
        "plt.ylabel(\"JPL SMAPE\")\n",
        "plt.grid(True)\n",
        "plt.show()\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 581
        },
        "id": "9OsKSalbJNJc",
        "outputId": "a24371b9-9c89-4a73-a8b0-e3559ca85795"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.24806302262512678\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "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06RJk9SsWTOFh4erVq1aGjJkiM6ePeu03pdffqmuXbuqevXqioiIUL169TRo0KBr9sNisWj27Nm6ePGifTpi/rlTxf17vHz5cvvf46K+L9u2bdPdd9+tSpUqKTIyUu3bt9fmzZud1jly5IieeuopNWrUSBEREapWrZruv/9+l38rzp07p5EjR9r/psbGxuqxxx4rMG03Ly9Pr7/+umJjYxUeHq5OnTpp//791+yr5Doukuvz9FauXKnbb79dlStXVoUKFdSoUSO99NJLTutkZ2drzJgxatiwocLCwhQXF6fRo0cX+Gyv9f8KYBZGkgAPWLJkierXr6/bbrvNrfX/+Mc/6t///rf69u2rZ555Rtu2bdP48eO1Z88eLVq0yGnd/fv3Kzk5WUOGDNEjjzyit99+Wz179tSMGTP00ksv6amnnpIkjR8/Xg888ID27t2roKDffv+wWq26++671bZtW7311ltatmyZxowZoytXruiVV16xrzd58mTde++9evjhh5WTk6P58+fr/vvv19KlS9WjRw+nPq1Zs0Yff/yxhg0bpurVq7v8T1WSnnjiCaWmpmrYsGFq2rSpfvnlF23atEl79uzRH/7wB0nSnDlzNHDgQLVp00bjx4/XqVOnNHnyZG3evFk7duxwGtGxWq3q2rWrbr31Vr399ttatWqV/vGPf6hBgwZ68sknr/mZjx07VuPGjdNdd92lJ598Unv37tX06dO1fft2bd68WSEhIZo0aZI++OADLVq0SNOnT1eFChXUokWLIuPpjvzkplq1ate1nbp160qSPvjgA/31r391q8DAHXfcodjYWKWkpNhjvmDBAlWoUKFAbB3NmzdPDRo0UJs2bdS8eXNFRkbqo48+0nPPPedWXw8ePKjg4OACo3KuzsMKDQ11meA5Ks53dNOmTUpLS9NTTz2lihUr6p///Kf69Omjn376yR6DXbt2qUuXLqpRo4bGjh2rK1euaMyYMcUe/bla/sFtlSpVrms7V3v00Ue1YsUKrVy5Ur/73e/s7Var1eVnGhERofLly7vcVq1atdS+fXt9/PHH9h8G8i1YsEDBwcG6//77JdniuHjxYt1///2qV6+eTp06pXfffVft27fX999/f10jo46GDBli/3vw5z//WYcOHdKUKVO0Y8cO+z56+vRpe8xeeOEFVa5cWYcPHy5y6ujcuXM1c+ZM/e9//7NPZ8z/e12cv8d79+5Vv379NGTIED3++ONq1KhRoa+5Zs0adevWTa1bt9aYMWMUFBSk2bNn684779TGjRt1yy23SJK2b9+uL774Qg899JBiY2N1+PBhTZ8+XR06dND333+vyMhISbZzERMTE7Vnzx4NGjRIf/jDH3TmzBn997//1bFjx1S9enX7a7/55psKCgrSs88+q4yMDL311lt6+OGHtW3btuIHxoXdu3frnnvuUYsWLfTKK68oLCxM+/fvd0oA8/LydO+992rTpk3605/+pCZNmmjXrl2aOHGifvzxRy1evLjA5+XO/ytAqTEAXJeMjAxDktGrVy+31t+5c6chyfjjH//o1P7ss88akow1a9bY2+rWrWtIMr744gt72/Llyw1JRkREhHHkyBF7+7vvvmtIMtauXWtv69+/vyHJePrpp+1teXl5Ro8ePYzQ0FDj559/trdfunTJqT85OTlG8+bNjTvvvNOpXZIRFBRk7N69u8B7k2SMGTPGvlypUiVj6NChhX4WOTk5Rs2aNY3mzZsbly9ftrcvXbrUkGS8/PLLBd7LK6+84rSN3//+90br1q0LfQ3DMIzTp08boaGhRpcuXQyr1WpvnzJliiHJeP/99+1tY8aMMSQ5fTaFcbVu//79jfLlyxs///yz8fPPPxv79+833njjDcNisRgtWrQo0es4unTpktGoUSNDklG3bl1jwIABxqxZs4xTp05ds3/PPvus0bBhQ/tjbdq0MQYOHGgYhi1uV8cpJyfHqFatmvGXv/zF3pacnGy0bNmywOu0b9/eaNy4sf0979mzx/jzn/9sSDJ69uzp9NlIcnnr2rWrW+/96j4W9h0NDQ019u/fb2/75ptvDEnGO++8Y2/r3bu3ER4e7rQfff/990ZwcLDhzn+PrmL99ttvGxaLxWjevLmRl5dX5DZcba8wO3bsMCQZI0eOtLe1b9++0M90yJAh13y9/L8Zu3btcmpv2rSp02ealZXltN8YhmEcOnTICAsLc9ofDx06ZEgyZs+e7dS/9u3bu3yvdevWtS9v3LjRkGTMmzfPab1ly5Y5tS9atMiQZGzfvv2a780VV59vSf4eL1u2rMjXysvLM2688Uaja9euTt+DS5cuGfXq1TM6d+7s1Ha1LVu2GJKMDz74wN728ssvG5KMtLQ0l69nGIaxdu1aQ5LRpEkTIzs72/745MmTXcb6alfHJV/+35J8EydOLPLv19y5c42goCBj48aNTu0zZswwJBmbN2+2t13r/xXALEy3A65TZmamJKlixYpurf/ZZ59JkkaNGuXU/swzz0hSgXOXmjZtqoSEBPvyrbfeKkm68847dcMNNxRodzXFx7FyWf60hpycHK1atcre7jiv/uzZs8rIyFBiYmKBqXGS1L59ezVt2rSId2o7r2fbtm1OFdccffnllzp9+rSeeuopp7nnPXr0UOPGjV2ex/XEE084LScmJhY5rWnVqlXKycnRiBEjnEbZHn/8cUVFRXn8fLGLFy+qRo0aqlGjhho2bKiXXnpJCQkJBX6VLomIiAht27bNPpozZ84cDR48WDExMXr66acLTGPJl5ycrP3792v79u32+2tNtfv888/1yy+/qF+/fva2fv366ZtvvtHu3bsLrP/DDz/Y33OTJk30zjvvqEePHnr//fed1gsPD9fKlSsL3ByLO1zrvecr6jt61113qUGDBvblFi1aKCoqyv5dsVqtWr58uXr37u20HzVp0kRdu3Ytsi/5ro71s88+q3bt2umTTz7xeBnxChUqSLJNY3UUHx/v8jMdMWLENbeXlJSkcuXKOZ3j9N133+n777/Xgw8+aG8LCwuz7zdWq1W//PKLfXqVq8++JBYuXKhKlSqpc+fOOnPmjP3WunVrVahQQWvXrpX027mCS5cuVW5u7nW/bnH/HterV8+t78fOnTu1b98+JScn65dffrG/n4sXL6pTp07asGGDvVqh4/c6NzdXv/zyixo2bKjKlSs7fb7/+c9/1LJlS913330FXu/q79rAgQOdzr1KTEyU5Pr/h5LIj8Mnn3zisuqiZItpkyZN1LhxY6eY3nnnnZJkj2k+d/9fAUoL0+2A65Q/RejqA5fCHDlyREFBQQUqp0VHR6ty5co6cuSIU7vjAZz0W8WquLg4l+1Xz98PCgpS/fr1ndryp+o4znlfunSpXnvtNe3cudPpQNvVgV69evUKfX+O3nrrLfXv319xcXFq3bq1unfvrscee8zen/z36mrKSuPGjbVp0yantvDwcNWoUcOprUqVKgXe89UKe53Q0FDVr1+/wGd+vcLDw7VkyRJJtgPMevXqKTY21mPbr1Spkt566y299dZbOnLkiFavXq23335bU6ZMUaVKlfTaa68VeM7vf/97NW7cWCkpKapcubKio6PtByuufPjhh6pXr559Go0kNWjQQJGRkZo3b16B8snx8fH617/+ZS/nfeONN6pmzZoFthscHKy77rqrRO+7ON/Rq/cbyfm78vPPP+vy5cu68cYbC6zXqFEj+8FzURxjfezYMb311lv2AgGeduHCBUkFf5ApX758iT7T6tWrq1OnTvr444/16quvSrJNtStXrpySkpLs6+VXe5w2bZoOHTrkdA7g9U4fzbdv3z5lZGS4/M5IvxVUad++vfr06aNx48Zp4sSJ6tChg3r37q3k5GSFhYUV+3WL+/fY3b99+/btkyT179+/0HUyMjJUpUoVXb58WePHj9fs2bN1/Phxp/NFMzIy7P8+cOCA+vTp49brX/39z5/6WdTfSnc9+OCDeu+99/THP/5RL7zwgjp16qSkpCT17dvXnlDv27dPe/bsKfA3O19+TPO5+9kCpYUkCbhOUVFRql27tr777rtiPc/dX5kLK41cWLtxVUEGd2zcuFH33nuv7rjjDk2bNk0xMTEKCQnR7NmzlZKSUmB9dw8AH3jgASUmJmrRokVasWKF/v73v+tvf/ub0tLS1K1bt2L301/KRF9PIlBcdevW1aBBg3Tfffepfv36mjdvnsskSbKNJk2fPl0VK1bUgw8+6DSq5igzM1NLlixRVlaWyyQiJSVFr7/+utN3uKQH6u4q7nfUk/vHtVwd665du6px48YaMmSI/vvf/3r0tfL/xpS0NL0rDz30kAYOHKidO3eqVatW+vjjj9WpUyen81veeOMN/d///Z8GDRqkV199VVWrVlVQUJBGjBhR6ChCPovF4vIzv7rYSl5enmrWrFlo2fT8A+38C6Vu3bpVS5Ys0fLlyzVo0CD94x//0NatW+2jbcXl7t9jd//25X8uf//73wu9aG1+X59++mnNnj1bI0aMUEJCgipVqiSLxaKHHnqoyM+3MCX9/hf2OVwdr4iICG3YsEFr167Vp59+qmXLlmnBggW68847tWLFCgUHBysvL0833XSTJkyY4HKbV//QR5VA+BqSJMAD7rnnHs2cOVNbtmxxmhrnSt26dZWXl6d9+/apSZMm9vZTp07p3Llz9hPzPSUvL08HDx50OtE7vzpW/omx//nPfxQeHq7ly5c7/Ro7e/bs6379mJgYPfXUU3rqqad0+vRp/eEPf9Drr7+ubt262d/r3r17C4xq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tj//+c+e7R0AADCd1Wq7+G56uq10emIio3IAAptbSdINN9ygf/3rX05t0dHRmjt3rn3ZYrGQJAEAEGDS0qThw6Vjx35ri421VRDk/C4AgcqtJOnw4cNe7gYAAPA1aWm2UupXX2/q+HFbO5UCAQQqt85JKsq5c+c0ZcoUT2wKAAD4AKvVNoLk6oK8+W0jRtjWA4BAc11J0urVq5WcnKyYmBiNGTPGU30CAAAm27jReYrd1QxDOnrUth4ABJpiJ0lHjx7VK6+8onr16qlLly6yWCxatGiRTp486Y3+AQAAE6Sne3Y9APAnbiVJubm5Wrhwobp27apGjRpp586d+vvf/66goCD95S9/0d13362QkBBv9xUAAJSSmBjPrgcA/sStwg116tRR48aN9cgjj2j+/PmqUqWKJKlfv35e7RwAADBHYqKtit3x467PS7JYbI8nJpZ+3wDA29waSbpy5YosFossFouCuTACAAABLzjYVuZbsiVEjvKXJ03iekkAApNbSdKJEyf0pz/9SR999JGio6PVp08fLVq0SJar/2oCAICAkZRkK/Ndp45ze2ws5b8BBDa3kqTw8HA9/PDDWrNmjXbt2qUmTZroz3/+s65cuaLXX39dK1eulJUaoAAABJykJOnwYWntWiklxXZ/6BAJEoDAVuzqdg0aNNBrr72mI0eO6NNPP1V2drbuuece1apVyxv9AwAAJgsOljp0kPr1s90zxQ5AoHOrcIMrQUFB6tatm7p166aff/5Zc+fO9WS/AAAAAMAU13Ux2Xw1atTQqFGjiv288ePHq02bNqpYsaJq1qyp3r17a+/evU7rZGVlaejQoapWrZoqVKigPn366NSpU57oNgAAAAAU4NZIUv369d3a2MGDB4v14uvXr9fQoUPVpk0bXblyRS+99JK6dOmi77//XuXLl5ckjRw5Up9++qkWLlyoSpUqadiwYUpKStLmzZuL9VoAAAAA4A63kqTDhw+rbt26Sk5OVs2aNT324suWLXNanjNnjmrWrKmvvvpKd9xxhzIyMjRr1iylpKTozjvvlCTNnj1bTZo00datW9W2bVuP9QUAAAAAJDeTpAULFuj999/XhAkT1K1bNw0aNEjdu3dXUJBHZuvZZWRkSJKqVq0qSfrqq6+Um5uru+66y75O48aNdcMNN2jLli0uk6Ts7GxlZ2fblzMzMyVJubm5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+uMf/6jly5d7uaeBq7gxyLd3716lp6fbbzVr1vRSD8uG9evXa+jQodq6datWrlyp3NxcdenSRRcvXiz0OV988YX69eunwYMHa8eOHerdu7d69+6t7777rhR7HjhKEgPJdpV1x33hyJEjpdTjwBQbG6s333xTX331lb788kvdeeed6tWrl3bv3u1yffYDzytuDCT2A2/bvn273n33XbVo0eKa67E/eI+7MZDYH0xhBDBJxqJFi665zujRo41mzZo5tT344ING165dvdizssOdGKxdu9aQZJw9e7ZU+lRWnT592pBkrF+/vtB1HnjgAaNHjx5ObbfeeqsxZMgQb3evTHAnBrNnzzYqVapUep0qo6pUqWK89957Lh9jPygd14oB+4F3nT9/3rjxxhuNlStXGu3btzeGDx9e6LrsD95RnBiwP5gj4EaSimvLli266667nNq6du2qLVu2mNSjsqtVq1aKiYlR586dtXnzZrO7E3AyMjIkSVWrVi10HfYH73InBpJ04cIF1a1bV3FxcUX+2o7isVqtmj9/vi5evKiEhASX67AfeJc7MZDYD7xp6NCh6tGjR4HvuSvsD95RnBhI7A9mKGd2B8x28uRJ1apVy6mtVq1ayszM1OXLlxUREWFSz8qOmJgYzZgxQzfffLOys7P13nvvqUOHDtq2bZv+8Ic/mN29gJCXl6cRI0aoXbt2at68eaHrFbY/cH7Y9XM3Bo0aNdL777+vFi1aKCMjQ2+//bZuu+027d69W7GxsaXY48Cya9cuJSQkKCsrSxUqVNCiRYvUtGlTl+uyH3hHcWLAfuA98+fP19dff63t27e7tT77g+cVNwbsD+Yo80kSzNeoUSM1atTIvnzbbbfpwIEDmjhxoubOnWtizwLH0KFD9d1332nTpk1md6XMcjcGCQkJTr+u33bbbWrSpIneffddvfrqq97uZsBq1KiRdu7cqYyMDKWmpqp///5av359oQfp8LzixID9wDuOHj2q4cOHa+XKlZz4b5KSxID9wRxlPkmKjo7WqVOnnNpOnTqlqKgoRpFMdMstt3BA7yHDhg3T0qVLtWHDhiJ/cSpsf4iOjvZmFwNecWJwtZCQEP3+97/X/v37vdS7siE0NFQNGzaUJLVu3Vrbt2/X5MmT9e677xZYl/3AO4oTg6uxH3jGV199pdOnTzvN0rBardqwYYOmTJmi7OxsBQcHOz2H/cGzShKDq7E/lI4yf05SQkKCVq9e7dS2cuXKa86Thvft3LlTMTExZnfDrxmGoWHDhmnRokVas2aN6tWrV+Rz2B88qyQxuJrVatWuXbvYHzwsLy9P2dnZLh9jPygd14rB1dgPPKNTp07atWuXdu7cab/dfPPNevjhh7Vz506XB+fsD55Vkhhcjf2hlJhdOcLTzp8/b+zYscPYsWOHIcmYMGGCsWPHDuPIkSOGYRjGCy+8YDz66KP29Q8ePGhERkYazz33nLFnzx5j6tSpRnBwsLFs2TKz3oLfK24MJk6caCxevNjYt2+fsWvXLmP48OFGUFCQsWrVKrPeQkB48sknjUqVKhnr1q0z0tPT7bdLly7Z13n00UeNF154wb68efNmo1y5csbbb79t7NmzxxgzZowREhJi7Nq1y4y34PdKEoNx48YZy5cvNw4cOGB89dVXxkMPPWSEh4cbu3fvNuMtBIQXXnjBWL9+vXHo0CHj22+/NV544QXDYrEYK1asMAyD/aA0FDcG7Ael5+rKauwPpa+oGLA/mCPgkqT8ctJX3/r3728YhmH079/faN++fYHntGrVyggNDTXq169vzJ49u9T7HUiKG4O//e1vRoMGDYzw8HCjatWqRocOHYw1a9aY0/kA4ioGkpy+3+3bt7fHJd/HH39s/O53vzNCQ0ONZs2aGZ9++mnpdjyAlCQGI0aMMG644QYjNDTUqFWrltG9e3fj66+/Lv3OB5BBgwYZdevWNUJDQ40aNWoYnTp1sh+cGwb7QWkobgzYD0rP1Qfo7A+lr6gYsD+Yw2IYhlF641YAAAAA4NvK/DlJAAAAAOCIJAkAAAAAHJAkAQAAAIADkiQAAAAAcECSBAAAAAAOSJIAAAAAwAFJEgAAAAA4IEkCAAAAAAckSQAAAADggCQJAOAXBgwYIIvFIovFopCQENWrV0+jR49WVlaWW89fv3697rzzTlWtWlWRkZG68cYb1b9/f+Xk5EiS1q1bZ99+UFCQKlWqpN///vcaPXq00tPTvfnWAAA+hiQJAOA37r77bqWnp+vgwYOaOHGi3n33XY0ZM6bI533//fe6++67dfPNN2vDhg3atWuX3nnnHYWGhspqtTqtu3fvXp04cULbt2/X888/r1WrVql58+batWuXt94WAMDHWAzDMMzuBAAARRkwYIDOnTunxYsX29v69OmjQ4cO6euvv77mcydNmqTJkyfr0KFDha6zbt06dezYUWfPnlXlypXt7ZcvX9bvf/97Va9eXZs2bbretwEA8AOMJAEA/NJ3332nL774QqGhoUWuGx0drfT0dG3YsKHYrxMREaEnnnhCmzdv1unTp0vSVQCAnylndgcAAHDX0qVLVaFCBV25ckXZ2dkKCgrSlClTinze/fffr+XLl6t9+/aKjo5W27Zt1alTJz322GOKiooq8vmNGzeWJB0+fFg1a9a87vcBAPBtjCQBAPxGx44dtXPnTm3btk39+/fXwIED1adPnyKfFxwcrNmzZ+vYsWN66623VKdOHb3xxhtq1qyZW0UZ8memWyyW634PAADfR5IEAPAb5cuXV8OGDdWyZUu9//772rZtm2bNmuX28+vUqaNHH31UU6ZM0e7du5WVlaUZM2YU+bw9e/ZIkuLj40vadQCAHyFJAgD4paCgIL300kv661//qsuXLxf7+VWqVFFMTIwuXrx4zfUuX76smTNn6o477lCNGjVK2l0AgB8hSQIA+K37779fwcHBmjp16jXXe/fdd/Xkk09qxYoVOnDggHbv3q3nn39eu3fvVs+ePZ3WPX36tE6ePKl9+/Zp/vz5ateunc6cOaPp06d7860AAHwIhRsAAH6rXLlyGjZsmN566y09+eSTKl++vMv1brnlFm3atElPPPGETpw4oQoVKqhZs2ZavHix2rdv77Ruo0aNZLFYVKFCBdWvX19dunTRqFGjFB0dXRpvCQDgA7hOEgAAAAA4YLodAAAAADggSQIA+L033nhDFSpUcHnr1q2b2d0DAPgZptsBAPzer7/+ql9//dXlYxEREapTp04p9wgA4M9IkgAAAADAAdPtAAAAAMABSRIAAAAAOCBJAgAAAAAHJEkAAAAA4IAkCQAAAAAckCQBAAAAgAOSJAAAAABw8P8AfCAf0A9yvPUAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sparsity={'169': 0.9779411764705882,\n",
        " '171': 0.9656862745098039,\n",
        " '176': 0.9754901960784313,\n",
        " '220': 0.9558823529411765,\n",
        " '322': 0.9705882352941176,\n",
        " '334': 0.9901960784313726,\n",
        " '335': 0.9583333333333334,\n",
        " '346': 0.9436274509803921,\n",
        " '365': 0.9509803921568627,\n",
        " '368': 0.9656862745098039,\n",
        " '372': 0.9754901960784313,\n",
        " '374': 0.9607843137254902,\n",
        " '378': 0.9730392156862745,\n",
        " '382': 0.9583333333333334,\n",
        " '404': 0.9754901960784313,\n",
        " '405': 0.9362745098039216,\n",
        " '406': 0.9877450980392157,\n",
        " '409': 0.9779411764705882,\n",
        " '410': 0.9485294117647058,\n",
        " '416': 0.9583333333333334,\n",
        " '436': 0.9705882352941176,\n",
        " '444': 0.9730392156862745,\n",
        " '458': 0.9583333333333334,\n",
        " '467': 0.9632352941176471,\n",
        " '474': 0.9779411764705882,\n",
        " '476': 0.9656862745098039,\n",
        " '481': 0.9607843137254902,\n",
        " '483': 0.9754901960784313,\n",
        " '507': 0.9632352941176471,\n",
        " '526': 0.9779411764705882,\n",
        " '531': 0.9656862745098039,\n",
        " '537': 0.9705882352941176,\n",
        " '551': 0.9411764705882353,\n",
        " '553': 0.9828431372549019,\n",
        " '576': 0.9705882352941176,\n",
        " '577': 0.9583333333333334,\n",
        " '581': 0.9779411764705882,\n",
        " '592': 0.9730392156862745,\n",
        " '607': 0.9436274509803921,\n",
        " '651': 0.9656862745098039,\n",
        " '726': 0.9705882352941176,\n",
        " '742': 0.9607843137254902,\n",
        " '826': 0.9730392156862745,\n",
        " '933': 0.9632352941176471}\n",
        "\n",
        "entropy={'169': 2.0723243489301826,\n",
        " '171': 3.5841837197791895,\n",
        " '176': 2.924297799747892,\n",
        " '220': 3.9861879650463035,\n",
        " '322': 2.8868392966712704,\n",
        " '334': 1.5327896019567016,\n",
        " '335': 3.825251737288651,\n",
        " '346': 4.423251796980337,\n",
        " '365': 4.1852301329094015,\n",
        " '368': 3.313502741214098,\n",
        " '372': 2.8851038773309874,\n",
        " '374': 3.6889681813826796,\n",
        " '378': 3.0666152167384504,\n",
        " '382': 3.9754180179138334,\n",
        " '404': 3.021928094887363,\n",
        " '405': 4.430640983527282,\n",
        " '406': 1.1440181631019015,\n",
        " '409': 2.823219672335508,\n",
        " '410': 4.060574474528751,\n",
        " '416': 3.9754180179138334,\n",
        " '436': 3.2813734094119917,\n",
        " '444': 2.9796586949993205,\n",
        " '458': 3.9754180179138334,\n",
        " '467': 3.638147696204827,\n",
        " '474': 2.558810827984542,\n",
        " '476': 3.6074753914554893,\n",
        " '481': 3.8868421881310113,\n",
        " '483': 2.6875127440902498,\n",
        " '507': 3.641446071165522,\n",
        " '526': 2.393828022094786,\n",
        " '531': 3.383014003266002,\n",
        " '537': 3.101609497059027,\n",
        " '551': 4.487122805397797,\n",
        " '553': 2.511737433422467,\n",
        " '576': 3.2811939311696197,\n",
        " '577': 3.863465189601647,\n",
        " '581': 2.865764637179023,\n",
        " '592': 2.9742725050160947,\n",
        " '607': 4.365013648887185,\n",
        " '651': 3.3722539283649273,\n",
        " '726': 3.1876013115120565,\n",
        " '742': 3.7492750707107136,\n",
        " '826': 3.2841837197791888,\n",
        " '933': 3.794653473544342}\n",
        "\n",
        "\n",
        "# Convert the keys of R_SD to integers for proper comparison\n",
        "entropy_int_keys = {int(k): v for k, v in entropy.items()}\n",
        "\n",
        "# Finding common keys between the two dictionaries\n",
        "common_keys_ent = set(user_smape_JPL.keys()).intersection(entropy_int_keys.keys())\n",
        "x_values_ent = [entropy_int_keys[key] for key in common_keys_ent]\n",
        "y_values_ent = [user_smape_JPL[key] for key in common_keys_ent]\n",
        "\n",
        "correlation_coefficient_ent = np.corrcoef(x_values_ent, y_values_ent)[0, 1]\n",
        "print(correlation_coefficient_ent)\n",
        "# Calculating the line of best fit\n",
        "m, b = np.polyfit(x_values_ent, y_values_ent, 1)  # m is slope, b is y-intercept\n",
        "# Generating y-values for the line of best fit\n",
        "fit_line_ent= [m*x + b for x in x_values_ent]\n",
        "\n",
        "\n",
        "# Convert the keys of R_SD to integers for proper comparison\n",
        "sparsity_int_keys = {int(k): v for k, v in sparsity.items()}\n",
        "\n",
        "# Finding common keys between the two dictionaries\n",
        "common_keys_spars = set(user_smape_JPL.keys()).intersection(sparsity_int_keys.keys())\n",
        "x_values_spars = [sparsity_int_keys[key] for key in common_keys_spars]\n",
        "y_values_spars = [user_smape_JPL[key] for key in common_keys_spars]\n",
        "\n",
        "correlation_coefficient_spars = np.corrcoef(x_values_spars, y_values_spars)[0, 1]\n",
        "print(correlation_coefficient_spars)\n",
        "# Calculating the line of best fit\n",
        "m, b = np.polyfit(x_values_spars, y_values_spars, 1)  # m is slope, b is y-intercept\n",
        "# Generating y-values for the line of best fit\n",
        "fit_line_spars= [m*x + b for x in x_values_spars]\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5KZcZtbo1ZAo",
        "outputId": "522e948d-ae44-4f7a-b5c8-c2c9308cee22"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0.253086036144375\n",
            "-0.17484352496206504\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Create a subplot with 1 row and 2 columns\n",
        "fig, ax = plt.subplots(1, 2, figsize=(20, 6))\n",
        "\n",
        "# Plotting for Entropy_SD\n",
        "ax[0].scatter(x_values_ent, y_values_ent, color='blue')\n",
        "ax[0].plot(x_values_ent, fit_line_ent, color='black')\n",
        "ax[0].set_xlabel(\"Entropy_DE\",fontsize=16)\n",
        "ax[0].set_ylabel(\"SVR\",fontsize=16)\n",
        "ax[0].grid(True)\n",
        "ax[0].text(0.05, 0.76, f'Correlation: {correlation_coefficient_ent:.3f}', transform=ax[0].transAxes, fontsize=14)\n",
        "\n",
        "# Plotting for Sparsity_SD\n",
        "ax[1].scatter(x_values_spars, y_values_spars, color='blue')\n",
        "ax[1].plot(x_values_spars, fit_line_spars, color='black')\n",
        "ax[1].set_xlabel(\"Sparsity_DE\", fontsize=16)\n",
        "ax[1].set_ylabel(\"SVR\", fontsize=16)\n",
        "ax[1].grid(True)\n",
        "ax[1].text(0.05, 0.76, f'Correlation: {correlation_coefficient_spars:.3f}', transform=ax[1].transAxes, fontsize=14)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 404
        },
        "id": "TfTjXuR41bxI",
        "outputId": "540f169f-17e9-4466-83a6-1ab699918345"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 2000x600 with 2 Axes>"
            ],
            "image/png": "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AABA0ZIUBwAAAAAAAKBoSYoDAAAAAAAAULQkxQEAAAAAAAAoWpLiAAAAAAAAABQtSXEAAAAAAAAAipakOAAAAAAAAABFS1KcvJJIJGLYsGEFvw0AgFJWXx8xbVrEbbelHuvrc90jAFrTvXt3MThQkhyvAkDpkRQvArNnz46TTjopttlmm1h//fWjZ8+esdVWW8UxxxwTU6dOzXX3sm7YsGGRSCRy3Y2MaGhoiKuvvjp22GGH6NmzZ2y22WZx9NFHx1tvvZV2Gx988EH8/ve/j8MPPzy++MUvRo8ePWLTTTeNww47LB566KFWf+emm26KRCLR5te0adOa1V++fHmcfvrpsd9++0X//v1jvfXWi379+sU+++wTN954Y9TV1a3L2wAAFJCamojKyojhwyPGjk09VlamygGKgRi8uWKKwdfmT3/6U+yxxx6x/vrrx0YbbRRf+9rXYs6cOe1q44knnogzzjgjhg8fHn379o1EIhHHH398m/Ub3981fd1yyy3NfqeysrLNuk5QAMerAFCquua6A3RcQ0NDnHHGGXHllVdG165d44ADDojDDz88unXrFm+99Vbcd999ceutt8bPf/7zOP/883Pd3bzxz3/+M3r16pXrbqTlv//7v+O6666Lr3zlK/HDH/4w3n333fjLX/4SDz/8cMyaNSu22WabtbZxxx13xPe///3o379/HHjggTFgwICYP39+3HnnnfHggw/GL3/5y/jJT37S6u+OHj06dt555xbllZWVzX5eunRpXHvttbHHHnvEyJEjY7PNNouPPvooHnjggTjxxBPj9ttvjwceeCC6dHEeDgAUs5qaiDFjIpLJ5uULFqTKp0yJqK7OTd8A1pUYvGMKKQZfk4svvjjOO++8GDRoUHzve9+LTz75JG6//fbYe++949FHH4199tknrXZuuOGGuPnmm6NXr16x5ZZbxpIlS9ZY//jjj281kV1XVxeXXHJJdOnSJQ488MAWz/ft2zfGjx/fovzz8TyUGserAFC6JMUL2HnnnRdXXnll7LzzzjFlypTYaqutmj3/2WefxW9+85v44IMPctTD/DR48OBcdyEtjz/+eFx33XWx3377xdSpU6N79+4RETF27NgYMWJEnHrqqW1e6b26bbfdNv72t7/FyJEjmyWlzzvvvNhzzz3j3HPPjW9961vRv3//Fr97xBFHrPGM9UYbb7xxLF68uKmPjVatWhUHH3xwPPzww/HAAw/EyJEj19oWAFCY6usjxo1rOcEYkSpLJCLGj48YPTqirCzr3QNYZ2LwjimUGHxNXn/99ZgwYUJsu+228X//93/Rt2/fiIj4wQ9+EHvttVd85zvfiZdeeimtE8FPPfXU+MlPfhKDBw+OZ599NoYOHbrG+m3F5HfeeWckk8kYMWJEq/H8hhtuGBMmTFhrf6CUOF4FgNLmss0C9cYbb8Qvf/nL2GSTTeLBBx9sEYxHRPTs2TN+8pOfxMSJE5uVL1q0KMaPHx9f+MIXokePHrH55pvHN7/5zXjppZdatHH88cdHIpGIt956K/7nf/4nvvzlL0ePHj2agrLKysqorKyMjz/+OE499dQYOHBgdO3aNW666aamNl544YU46qijory8PLp37x6DBg2K0047Le2Jgtdeey3OPPPM2HXXXWOTTTaJ9dZbL7bddts4++yzY+nSpc3qJhKJmD59etP3jV+rB5FtLRfWkfdl7ty58etf/zoGDx4cPXr0iEGDBsXEiROjoaEhrde2Jn/4wx8iIuLCCy9slmw+7LDDYtiwYfHwww/HO++8s9Z2DjjggBg1alSL4Hy77baL//qv/4q6urp4+umn16mvXbp0aZEQj4jo2rVrfP3rX4+I1D4LABSvGTMi5s9v+/lkMmLevFQ9gEIjBi/+GHxNbrzxxli1alWce+65TQnxiIidd945jj766PjnP/8ZTz75ZFpt7bbbbvGVr3wlytYx43b99ddHRMRJJ520Tu1AKXG8CgClzZXiBeqmm26K+vr6+O///u/YYost1li3R48eTd+///77MXTo0HjzzTdj2LBhcdRRR8XcuXNjypQpcd9998VDDz0U++67b4s2TjvttJg1a1aMHDkyRo0aFZtvvnnTcytWrIgDDjggli5dGocffnh07dq1qU9/+9vf4pvf/GZ06dIlRo8eHQMHDoyXX345fvOb38RDDz0UzzzzTGy00UZr7H9NTU1cf/31MXz48Bg2bFg0NDTErFmz4rLLLovp06fHE088Ed26dYuIiAsuuCBuuumm+Ne//hUXXHBBUxutLQG+usWLF0dVVVW735ef/OQnMX369Pja174Whx56aNx9990xYcKEWLlyZVx88cXN6lZWVsa//vWvmDt3blrLlU2bNi3WX3/9VpdgO/TQQ2PatGkxffr0OOaYY9baVlsa37euXVv/U/Dcc8/FBx98EKtWrYrKyso46KCDYpNNNkm7/YaGhnjwwQcjImL77bfvcD8BgPxXW5vZegD5RAye2Ri8o+9LZ8bgazJt2rSIiDjkkENaPHfooYfGTTfdFNOnT4/99ttvnbaTrvnz58dDDz0U5eXlba7ItmLFirjpppvi3XffjT59+sTuu+8ee+65Z1b6B/nK8SoAlDZJ8QL11FNPRUTqKuD2OOuss+LNN9+Mc845J37xi180ld9///0xcuTIOOGEE+LVV19tcVXxCy+8EM8991xsueWWLdpcuHBh7LTTTvHUU09Fz549m8o/+OCDOOaYY2LTTTeNp556KgYNGtT03O233x5HH310/OxnP4urr756jX0+5phj4vTTT29xJfLPf/7zuOCCC+Ivf/lLfOtb34qIiAkTJsS0adPiX//6V7uWCfvjH//Yofdlzpw58cILL0R5eXlERJx//vmxzTbbxNVXXx0XXHBBq1dPp+PTTz+N2tra2H777Vs9e7zxXuKvv/56h9qPiFiyZElMmTIl1ltvvaiqqmq1zq9//etmP/fs2TMuuOCCOOuss1qtv3LlyvjFL34RyWQyPvjgg3j00UfjlVdeiRNOOKHVe5wBAMXj/x0OZaweQD4Rg2c2Bu/o+9JZMfjavP7669G7d+/o169fi+cyEZ+314033hgNDQ1x3HHHtXmS+8KFC+OEE05oVrb77rvHbbfd1upKB1AKHK8CQGmzfHqBWrhwYUREVFRUpP07K1eujNtuuy022WSTOO+885o9N2LEiDj44IPjjTfeaAr2V/eTn/yk1WC80S9/+ctmwXhEKtG8ZMmSuOSSS5oF4xERRx11VOy6665x++23r7XfAwYMaDWwPfXUUyMi4pFHHllrG2uycuXKmDFjRofel/PPP78pGI+I2HTTTWP06NHxySefxKuvvtqs7qOPPhr//Oc/Y8CAAWvt0+LFiyMimi3Ltro+ffo0q9cR3/ve9+K9996Ln/70py2u/v7CF74QV199dbz22muxbNmymD9/fvzxj3+MjTfeOM4+++w2J1FWrlwZEydOjJ///OdxzTXXxKuvvhpnnHFG/P73v+9wPwGAwlBVFVFRkboXY2sSiYiBA1P1AAqNGDyzMXhH35fOisHXZvHixZ0an7dHMpmMG2+8MSLaXjr9hBNOiEcffTTee++9+PTTT+O5556LY445Jp599tk48MAD45NPPslKXyHfOF4FgNLmSvES8sorr8Ty5ctj+PDh0atXrxbPDx8+PKZOnRrPP/98iyuH99hjjzbbXW+99WKHHXZoUT5r1qyIiHjmmWfizTffbPH88uXLY9GiRbFo0aLYdNNN22y/MeC76aab4qWXXorFixc3u1/Yu+++2+bvpuOVV16JlStXxu67797u92XIkCEt6jdOknz88cfNyvPpTOxzzjknbrvttvjqV78aP/3pT1s8v//++8f+++/f9POAAQPimGOOiV133TV22223mDBhQnz/+99vcUZ67969I5lMRkNDQ7z77rtxzz33xE9/+tOYOXNm3H///U2TBQBA8Skri5g0KWLMmNSEYjL5n+caJx6vuipVD6AU5HMM3laCNyI7MXhH35fOisGff/75uPvuu5uVVVZWNrs3er547LHHYu7cubH//vvH1ltv3Wqd1Zeyj0gtZ//HP/4xIiJuueWW+MMf/hCnn356p/cV8o3jVQAobZLiBapfv37xyiuvxIIFC2K77bZL63eWLFkSEdHm/c8az7ZurLe6Nd0zbfPNN49EK6dYfvjhhxERcc0116yxX59++ukak+I//OEP4ze/+U0MHDgwDj/88CgvL2+6R9vEiRNjxYoVa2x/bRrPkF79Hm2rW9P70lqStzFRXF9f3+E+NU5QtHWmeWNf1jSR0Zbzzz8/Lr300jjggAOipqam1eXZ2/KVr3wl9t1333jkkUfin//8Z6sTMRERXbp0iYqKivj+978fm266aXzzm9+Miy++OC677LJ29xcAKBzV1RFTpkSMGxcxf/5/yisqUhOM1dU56xrAOim2GHxNsWRnx+Dr8r50Vgz+/PPPx8SJE5uV7b///k1J8b59+3ZKfN4R119/fUREnHzyye3+3f/+7/+OW265JZ566ilJcUqW41UAKF2S4gVqn332iWnTpsWjjz6a9j3NGoPH9957r9XnG5eDay3IbC3gXttzje28+OKLsf3226fVx8/797//Hddcc03suOOOMXPmzGZnkS9cuLBF0NoRG2ywQdO2WrOm96WzrL/++lFeXh5z586N+vr6FonrxnuVNd67LF3nn39+XHTRRTFs2LC45557Wiy3l47GExg+/fTTtOofcsghERExbdq0dm8LACg81dURo0dHzJgRUVubuidjVZUrboDCVmwxeF1dXavl2YjB1+V96SzHH3/8Gq8K32abbWLmzJmxcOHCFvcV72h83hEfffRR3HXXXbHhhhvGmDFj2v377Y3noVg5XgWA0uSe4gXq+OOPj7Kysvj9738f77///hrrNp7FPXjw4FhvvfXi2WefjWXLlrWo15i03HnnnTPSxz333DMiImbOnNnhNt56661IJpNx0EEHtVhWbcaMGa3+TmMCOd2zxAcPHhzdu3ePv//971l5X9K1//77x6efftrqfdQeeuihiIjYb7/90m6vMSG+//77x3333dfqMnVrU19fH3//+98jIlrco64tjUvrdevWrd3bAwAKU1lZxLBhEUcfnXo0wQgUOjF4ZmPwbL4vmdB4e7GHH364xXON8fnqtyDrLLfeemssX748vvWtb8V6663X7t9/5plnIiK1NDyUOserAFB6JMUL1NZbbx1nnnlmLFq0KA477LCYO3duizrLly+PK664IiZMmBAREd27d4+jjz46Fi1aFJdcckmzug8++GA89NBDsfXWW8c+++yTkT6ecMIJscEGG8S5554b//jHP1o8v2zZsqZ7nrWlMfH69NNPN7uH2fz58+Occ85p9Xc23njjiIiYN29eWv3s3r17VFVVdfr78uabb8Yrr7zS5hn5n/fd7343IlLJ7JUrVzaVP/DAAzFt2rQ45JBDWiSmX3nllXjllVdatPWzn/0sLrrooqiqqkorIT579uwWZfX19XH22WfHG2+8EcOHD29a0i4i4uWXX251MmPZsmVNS7KNGDFijdsEAADIV2LwzMbg2Xhf2huDr8kJJ5wQXbt2jYsvvrjZMurPP/983HbbbfGlL30p9t13307bfqPGpdNPOumkNuu88sorrcbnr7zySpx11lkRETF27NiM9QkAAAqF5dML2EUXXRTLly+PK6+8Mrbbbrs44IADYvvtt49u3brF3Llz45FHHokPPvggLrrooqbfueyyy2L69Olx0UUXxdNPPx177rlnvP3223HHHXdEr1694sYbb4wuXTJzrsRmm20Wt912W3zjG9+InXbaKb761a/G4MGDY8WKFfH222/H9OnTY++9944HH3ywzTbKy8vjyCOPjDvvvDN22223OPDAA+O9996Le++9Nw488MB48803W/zOAQccEFOmTIkjjzwyDjvssFhvvfVip512ilGjRrW5nWOPPTbmzp3bqe/LgQceGP/6179i7ty5aZ2VPXz48Dj55JPjuuuui1133TVGjhwZtbW18ec//zk23njjuPrqq1v8zpe+9KWIiEgmk01lN910U1x44YXRtWvX2GOPPeLyyy9v8XvDhg2LYcOGNf282267xY477hg77rhjDBgwID788MOYPn16vPbaa1FRURHXXXdds9//y1/+EldccUXsu+++UVlZGX369IkFCxbEAw88EB988EFUVVXFj370ozTfKQAAgPwjBs9cDJ6N96W9MfiabLvttjFhwoQ477zzYqeddoojjzwyPvnkk7j99tsjIuIPf/hDi/62tf0nn3yyKaZuXHXgySefbFq+fdNNN41f/epXLfowe/bs+P/+v/8vdt1119hll13a7Ovtt98eV1xxRey3334xaNCgWH/99eO1116L+++/P+rq6uKcc85p16pzAABQLCTFC1iXLl3iiiuuiLFjx8a1114bTzzxRDzxxBPR0NAQ5eXlceihh8YJJ5wQBx10UNPvbLbZZvHMM8/EhRdeGH/9619jxowZ0bdv3zjiiCPiggsu6PC9v9sycuTIeO655+Lyyy+PRx55JKZOnRrrr79+VFRUxAknnBDf/va319rGTTfdFJWVlXHnnXfG1VdfHVtuuWWcfvrpcdZZZ8WUKVNa1P/Od74Tb7/9dtx+++1x2WWXxapVq+K4445bY0Det2/fePLJJ+PSSy/NyvuSrt/97nexww47xO9///uYNGlS9O7dO77+9a/HxRdfHFtttVVabbz99tsREbFq1ar4n//5nzbrrZ4U//GPfxyzZs2KqVOnxocffhjdu3ePrbfeOs4777w4/fTTY6ONNmr2u1/72tfi3XffjaeffjpmzpwZS5cujb59+8aOO+4YRx11VJx44onRtas/NwAAQOESg2cuBs/2+5IJ5557blRWVsZVV10V1157bdOqcxdeeGHsuuuuabfzxhtvxM0339ys7M0332w64WDQoEGtJsUbrxI/+eST19j+8OHD45///Gc899xzMWPGjFi2bFlsuummMWLEiPjBD34QhxxySNp9BQCAYpJIrn5JaQlZsmRJ9O3bNxYvXhx9+vRpKq+rq4v7778/RowY4R7IJcKYlybjXpqMe2ky7qXHmJcm416a8nXc24o3S5X3I//k62eH4mNfI5vsb2ST/Y1ssr+RTYW4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tgJyAIA8Vl0d8fbbEY8/HjF5cupx7lwBMEA+EX8DAJBp5gOAUtauK8WXLFkSX/ziF1t9bquttoqIiJ122mndewUAQKcqK4sYNizXvQCgLeJvAAA6g/kAoFS160rxhoaG6NatW6vPde2ayq/36tVr3XsFAAAAJUz8DQAAAJnTrqQ4AAAAAAAAABSSdi2fHhHx4IMPxgEHHNDu5xOJRDz66KNpbeOSSy6JmpqaeOWVV6Jnz56x9957x2WXXRbbbbddU53ly5fHj3/847j99ttjxYoVceihh8Zvf/vb2GKLLdr7kgAAACDviL8BAAAgM9qdFF+4cGEsXLiw3c8nEom0tzF9+vQ45ZRTYvfdd49Vq1bFT3/60zjkkEPi5ZdfjvXXXz8iIn70ox/FfffdF3fccUf07ds3Tj311Kiuro6nnnqqvS8JAAAA8o74GwAAADKjXUnxCy64oLP60cyDDz7Y7OebbropNt9885g9e3bst99+sXjx4rj++utj8uTJTWfF33jjjfGlL30pZs2aFXvttVdW+gkAAACdQfwNAAAAmZOXSfHPW7x4cUREbLzxxhERMXv27Kirq4uDDjqoqc7gwYNjyy23jJkzZ7YalK9YsSJWrFjR9POSJUsiIqKuri7q6uqayhu/X72M4mbMS5NxL03GvTQZ99JjzEuTcS9N+TrumehPKcTf5E6+fnYoPvY1ssn+RjbZ38gm+xvZVIj7W7p9TSSTyWS6ja5cuTK6d+/e4U51RENDQxx++OHx8ccfx5NPPhkREZMnT44TTjihWZAdEbHHHnvE8OHD47LLLmvRzoQJE2LixIktyidPnhy9evXqnM4DAABQcpYtWxZjx46NxYsXR58+fTrUhvgbAAAA1i7dGLxdV4r3798/vv3tb8eJJ54YO+644zp3Mh2nnHJKvPTSS00BeUedc845cfrppzf9vGTJkhg4cGAccsghzd6gurq6mDp1ahx88MHRrVu3ddomhcGYlybjXpqMe2ky7qXHmJcm416a8nXcG6+MXhelEH+TO/n62aH42NfIJvsb2WR/I5vsb2RTIe5v6cbg7UqKf/jhh3H11VfH1VdfHbvuumucdNJJMXbs2E4Lak899dS4995744knnoiKioqm8n79+sXKlSvj448/jg033LCp/L333ot+/fq12laPHj2iR48eLcq7devW6qC2VU7xMualybiXJuNemox76THmpcm4l6Z8G/dM9KWU4m9yx5iQLfY1ssn+RjbZ38gm+xvZVEj7W7r97NKeRu+8884YMWJElJWVxezZs+OUU06J8vLyOPbYY+Pxxx/vUEdbk0wm49RTT4277rorHnvssfjCF77Q7PkhQ4ZEt27d4tFHH20qe/XVV+Odd96JoUOHZqwfAAAAkAvibwAAAMicdiXFv/71r8c999wT8+bNi0svvTS22267+Oyzz+LWW2+Ngw46KLbeeuv4xS9+EQsWLFinTp1yyilx6623xuTJk2ODDTaIhQsXxsKFC+Ozzz6LiIi+ffvGSSedFKeffno8/vjjMXv27DjhhBNi6NChsddee63TtgEAACDXxN8AAACQOe1KijfaYost4swzz4yXX345nnrqqTjppJOid+/e8dZbb8X5558flZWVMXLkyKipqYlVq1a1u/1rr702Fi9eHMOGDYvy8vKmrz//+c9Nda688sr42te+FkceeWTst99+0a9fv6ipqenIywEAAIC8JP4GAACAddehpPjqhg4dGn/4wx9i4cKFceONN8a+++4bDQ0N8cADD8Q3vvGN6N+/f5xxxhnxj3/8I+02k8lkq1/HH398U5311lsvrrnmmvjwww/j008/jZqamjbvZwYAAACFTvwNAAAAHbPOSfFGPXv2jOOOOy6mT58er7/+evz0pz+NAQMGxKJFi+LKK6+MnXbaKVObAgAAgJIl/gYAAID2yVhSfHVf/OIX4/zzz4+LLrooNttss6YzzQEAAIDMEX8DAADA2nXNdIP/93//FzfeeGPcfvvtsWTJkkgmk1FWVhYjR47M9KYAAACgZIm/AQAAID0ZSYq///77ccstt8SNN94YL7/8ckSk7ku2zTbbxIknnhjHHXec+40BAADAOhJ/AwAAQPt1OCne0NAQ9913X9xwww1x//33x6pVqyKZTEbPnj1jzJgxcdJJJ8V+++2Xyb4CAABAyRF/AwAAwLppd1L8lVdeiRtuuCFuvfXWeO+995ruVTZkyJA4+eST4+ijj44+ffpkvKMAAABQSsTfAAAAkBntSorvvffe8cwzz0REanm2jTfeOL71rW/FSSedFDvuuGOndBAAAABKjfgbAAAAMqddSfFZs2ZFIpGIAw44IE466aSorq6O7t27d1bfAAAAoCSJvwEAACBz2pUUP//88+OEE06I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pDbr1dfXxxtvvBFz5syJ2bNnNyXRlyxZkva2FixYEAsWLIi//e1va6y38cYbNyXNhwwZEkOGDIkvfvGLkucAdDrxHaXK/CUUJ0lxALJKQAXFrawsYtKkiDFjUp/n1T/njXP3V12VqgfQ6N13340HHnigKfG9fPnyNuvOnDkzampq4pRTTsliD6G5srKy2G677WK77baLo48+us16yWQy3nnnnWZXnc+ZMycWLlyY9rY+/PDDeOSRR+KRRx5ZY71evXq1WLb9S1/6UnTtauoHgI4R31GqzF9CcRIZAZBVAiooftXVEVOmRIwb1/weXBUVqc93dXXOugbk0MyZM2Pvvfde53YOOeSQOP7449e9Q5AFiUQiBg0aFIMGDYqvf/3ra6z773//u+nK88bk+dtvv532tpYtWxZPPvlkPPnkk2utu8suu8Smm24aCxYsiN133z122GGHWG+99dLeFgClQ3xHKTJ/CcVJUhyArBNQQfGrro4YPTpixozUPbbKy1NnUAsYobi98sorscsuu6zxSu90rL/++jFy5MgYMWJEHHroodGvX78M9RDy1+abbx5f/epX46tf/eoa6y1evDief/75psT57Nmz45///Ge7tvXcc89FRMTUqVPXWO/LX/5ys6Xbd9lll9hggw3atS0ACp/4jlJk/hKKT14mxZ944om4/PLLY/bs2VFbWxt33XVXHHHEEU3PH3/88XHzzTc3+51DDz00HnzwwSz3FICOElBB8Ssrixg2LNe9ADLtzTffjK233jojbX31q1+N0aNHx2GHHRaDBg3KSJu0nxi8sPTt2zf233//2H///ddY77PPPosXXnihxdLt7fHyyy/Hyy+/HLfeemta9Xv37h3f+c534uSTT44vf/nL7doWAPlNfEcpMn8JxSUvk+Kffvpp7LTTTnHiiSdGdRun23z1q1+NG2+8sennHj16ZKt7AGSIgAoA8tP7778fm2++eUbbvPLKK2P8+PEZbZPMEIMXp549e8aee+4Ze+65Z4vn6urq4v77748RI0ZERGqVh8arzhuXb+/Iig9Lly6NK6+8Mq688sq11j3uuOPi5JNPjn322ScSjeuQAgDkGfOXUDzyMil+2GGHxWGHHbbGOj169LCEHgAAQAd9+umnsd1228WCBQsy1uaZZ54Zl1xySXTp0iVjbdL5xOClrVu3brHDDjvEDjvsEMcdd1yb9RoaGuLNN99sdtX5I4880uHt3nzzzS1WIGjNqFGj4uSTT44RI0ZE1655OY0FAAAUgIKNJqZNmxabb755bLTRRnHAAQfERRddFJtsskmb9VesWBErVqxo+nnJkiURkTo7uq6urqm88fvVyyhuxrw0GffSZNxLk3EvPca8NBn31q1atSoOOuigePrppzPW5rbbbhvPPvts9OzZs9Xn6+vro76+PmPbW5N8Hfd8608mtCcGTzf+Jnc6+tmprKyMysrKNlcUiIhIJpPx97//Pa6//vq44YYb1qmfERH33HNP3HPPPWutt/fee8eJJ54YY8aMiV69eq3zdsmMfP07TXGyv5FN9jeyyf5GNhXi/pZuXxPJZDLZyX1ZJ4lEosX9zG6//fbo1atXfOELX4g333wzfvrTn0bv3r1j5syZUdbGzRwmTJgQEydObFE+efJkwRIAAFCwkslkXHXVVTF9+vSMtdmtW7e44YYbYoMNNshYm6Vk2bJlMXbs2Fi8eHH06dMn191pl0zE4OJv0jV//vx45JFH4pFHHomlS5dmbbubb755TJw4McrLy7O2TQAAoHOkG4MXZFL88956663Yaqut4pFHHokDDzyw1Tqtnak+cODAWLRoUbM3qK6uLqZOnRoHH3xwdOvWLWOvg/xlzEuTcS9Nxr00GffSY8xLUymM+4UXXhgXXnhhRtt8/fXXY9CgQRltM5vyddyXLFkSm266adEkxT9vbTF4uvE3uZOvn5221NbWxi233BI33HBDvPXWW1nd9uOPPx777LNPVrdZTAptX6Ow2d/IJvsb2WR/I5sKcX9LNwYv2OXTV/fFL34xNt1003jjjTfaTIr36NEjevTo0aK8W7durQ5qW+UUL2Nemox7aTLupcm4lx5jXpoKfdxvuOGGOOmkkzLa5nPPPRc777xzRtvMN/k27vnUl86wthi8vfE3uVMoY7LlllvGueeeG+eee+4a6y1ZsiRuv/32OOOMM+KTTz7JyLaHDx+eVr1bbrklvv3tb2dkm8WoUPY1ioP9jWyyv5FN9jeyqZD2t3T72aWT+5EV8+fPjw8++MCyVwAAQN7705/+FIlEotWvjibEH3744Ugmk61+FXtCnOwTg5Ov+vTpE9/97ndjyZIlbf5NTCaT8cknn8RRRx2V0W0fc8wxbf5tX/1rwoQJ0dDQkNFtAwAAa5eXSfGlS5fG888/H88//3xERMydOzeef/75eOedd2Lp0qXxk5/8JGbNmhVvv/12PProozF69OjYeuut49BDD81txwEAACJi2rRpbSZEOnol4c0339xmgufggw/O8CuglIjBKTW9e/eO2267bY2J82QyGatWrVrr1entNXHixCgrK1tr8vxb3/pWLFu2LKPbBgCAUpaXSfG///3vscsuu8Quu+wSERGnn3567LLLLvGzn/0sysrK4oUXXojDDz88tt122zjppJNiyJAhMWPGjFaXZwMAKET19RHTpkXcdlvqsb4+1z0CPu8f//hHm8mMdJfb/byLL764zeTMsccem+FXAClicGhdWVlZXHTRRWtNnieTybj++uszuu3JkyfH+uuvv9bk+dChQ2PhwoUZ3TYA605MD5B/8vKe4sOGDYtkMtnm8w899FAWewMAkF01NRHjxkXMn/+fsoqKiEmTIqqrO3/79fURM2ZE1NZGlJdHVFVFlJV1/nYhH9XW1kb//v0z2uawYcPisccei0QikdF2oaPE4LDuTjzxxDjxxBPXWu/RRx+Ngw46KGPbnTVrVlq3Mthkk01i2rRpsf3222ds20BpESemL9cxPQCty8srxQEASlVNTcSYMc2D54iIBQtS5TU1nb/9ysqI4cMjxo5NPVZWdv52IZcWL17c5hV4HU2I9+vXL+rq6lq9mvDxxx+XEAcoUQceeGBaV57/4x//iAEDBmRsux988EHssMMOad33/Pbbb8/YdoHiIE5MX65jegDaJikOAJAn6utTZ5O3drFeY9n48Z237JrgnWJWV1fX5uT/hhtu2OF2P/nkk1aTGbW1tdG1a14uzAVAAfjyl78c8+fPX2vy/L333othw4ZldNtHH310WsnzCy64YI2rTADFQZyYvlzH9ACsmaQ4AECemDGj5UTD6pLJiHnzUvUyTfBOMUgmk7HFFlu0OnHfvXv3DrdbW1vbZjKid+/eGXwFANA+m2++eTz++ONrTZ4vW7YsreXd2+PnP/95dOnSZY2J8+7du8cll1wSdXV1Gd02kB3ixPbJZUwPwNpJigMA5Ina2szWaw/BO4XkC1/4QquT7l//+tfjo48+6lCbs2fPbjOR0K9fvwy/AgDIrp49e8b111+/1uR5fX19fPe7383otp955plYf/3113rl+bbbbtvh/+NA5xAntk8uY3oA1k5S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tW1us802cdBBB8V+++3X5kVGFL90Y/CCu1K8o84555w4/fTTm35esmRJDBw4MA455JBmb1BdXV1MnTo1Dj744OjWrVsuukqWGfPSZNxL05rG/dxzz43LL788o9t7+eWXY+utt85om7Sfz3vpMealqT3j/uSTESNHrr3N++5zpXi+y9fPe+OV0aUq3fib3MnXzw7Zkc3/g/Y1ssn+RjbZ38imQtnfRowYEeedd95a682dOzduvvnmuPHGG6O2tnadtvn666/H66+/Htdee+0a633xi1+Mk046KY499tjYYost1mmbxa5Q9rfVpRuDF1xSvF+/fhER8d577zU7S/29996LnXfeuc3f69GjR/To0aNFebdu3Vod1LbKKV7GvDQZ99Lyu9/9Lk477bSMtvn000/H0KFDM9omncPnvfQY89KUzrjvt1/EJptELFjQ+v1UE4mIiopUvbKyTuooGZVvn/d86su66kgM3t74m9wxJqUpF/8H7Wtkk/2NbLK/kU3Fsr9tu+22cfHFF8fFF1+8xnrvv/9+3HLLLXHdddfFP//5z3Xa5ltvvRXnnntunHvuuWust9FGG8XJJ58cJ598cmy77bbrtM1CV0j7W7r97NLJ/ci4L3zhC9GvX7949NFHm8qWLFkSzzzzjKQEAPHXv/41EolEq18dTYg/+uijkUwmW/3yvweg8JSVRUyalPo+kWj+XOPPV10lIQ4RYnAoRv4PAgCFYLPNNovTTz89Xn755TbnZpPJZCxdujRuvvnm2DcDS7199NFHcfnll8d2223X5hxz49d3v/vd+L//+7/I87tUs5q8TIovXbo0nn/++Xj++ecjIrWUwvPPPx/vvPNOJBKJGD9+fFx00UXxt7/9LV588cU49thjo3///k33PAOguM2aNavNg5GO/i+49dZb2zywOuCAAzL7AgDIuerqiClTIgYMaF5eUZEqr67OTb8gF8TgUHr8HwQAisX6668fxx57bMyYMWONyfOVK1fGX//61xg1alRGtvuHP/wh9txzz+jSpcsak+fV1dXxwAMPRH19fUa2S8fl5fLpf//732P48OFNPzfei+y4446Lm266Kc4888z49NNP47vf/W58/PHHse+++8aDDz4Y6623Xq66DECGvfbaa7HddttltM1f/OIX8eUvfzlGjBhRMEu/ANB5qqsjRo+OmDEjorY2orw8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lZWlMWPGqHHjxqpUqZK6dOminTt3OvswKoRPJcXPnDmjtm3baubMmXZtn56ern79+qlnz57as2ePxowZoxEjRmjNmjVOjhQVydFyL3DgwAFlZmYW/oSFhTkpQlS01NRUjRo1Stu2bdO6det0/vx59e3bV2fOnClzny1btmjo0KG67777tHv3bt1666269dZb9eWXX7owclyO8pS7JFWvXr1IWz9y5IiLIkZFaNCggV588UXt2rVLX3zxhXr16qUBAwZo3759pW5PW/d8jpa5RDv3Njt37tTs2bPVpk2bi25He/ce9pa5RHuHb3r33Xc1duxYTZw4UWlpaWrbtq1iY2N14sSJUrd/8sknNXv2bL322mvav3+/HnroId12223avXt3iW0daX/wDc6qb1dddVWR8/emTZtccTiwOGfUt5MnT6pr164KDAzUp59+qv379+uVV15RrVq1XHVYsChn1LedO3cWObetW7dOkjR48GCXHBOsyRl17aWXXtKsWbP0+uuv66uvvtJLL72kKVOm6LXXXnPVYcGinFHfRowYoXXr1mnBggXau3ev+vbtqz59+igjI8NVh1V+xkdJMitWrLjoNuPGjTNXXXVVkWVDhgwxsbGxTowMzmRPuScnJxtJ5uTJky6JCc534sQJI8mkpqaWuc3tt99u+vXrV2RZp06dzIMPPujs8OAk9pT73LlzTY0aNVwXFFyiVq1a5u233y51HW3dO12szGnn3iUrK8u0aNHCrFu3zvTo0cMkJiaWuS3t3Ts4Uua0d/iq6OhoM2rUqMLfc3NzTWRkpJk8eXKp20dERJjXX3+9yLK4uDhz5513FlnmSPuD73BGfZs4caJp27atU+KFZ3NGffvrX/9qunXr5pyA4dGc9f/phRITE02zZs1MXl5exQQNj+SMutavXz9z7733XnQb+KaKrm9nz541/v7+5uOPPy6yTYcOHcwTTzxRwdFXPJ+6U9xRW7duVZ8+fYosi42N1datW90UEVypXbt2ioiI0PXXX6/Nmze7Oxxchl9//VWSVLt27TK3ob17H3vKXZJ+++03NW7cWA0bNrzk3aawttzcXC1dulRnzpxR586dS92Gtu5d7ClziXbuTUaNGqV+/fqVaMelob17B0fKXKK9w/ecO3dOu3btKtJG/Pz81KdPnzLPd9nZ2SWmFqhUqVKJO3MdbX/wfs6s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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}