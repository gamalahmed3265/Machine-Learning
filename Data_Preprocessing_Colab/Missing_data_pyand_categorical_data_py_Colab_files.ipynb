{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": " Missing_data .pyand categorical_data.py- Colab files.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "AZ2WadBmnm5H",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#import data by connecting to your google drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "D7wAVEyWrGEd",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "#Navigate to the file/data location in your drive and cd into it\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nUiJ5woLqtsB",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Missing_data.py"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "txPK07VEqt2X",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Data Preprocessing\n",
        "\n",
        "# Importing the libraries\n",
        "#import numpy as np\n",
        "#import matplotlib.pyplot as plt\n",
        "#import pandas as pd\n",
        "\n",
        "# Importing the dataset\n",
        "#dataset = pd.read_csv('Data.csv')\n",
        "#X = dataset.iloc[:, :-1].values\n",
        "#y = dataset.iloc[:, 3].values\n",
        "\n",
        "# Taking care of missing data\n",
        "#from sklearn.preprocessing import Imputer\n",
        "#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)\n",
        "#imputer = imputer.fit(X[:, 1:3])\n",
        "#X[:, 1:3] = imputer.transform(X[:, 1:3])\n",
        "\n",
        "# Data Preprocessing\n",
        "\n",
        "# Importing the libraries\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "\n",
        "# Importing the dataset\n",
        "dataset = pd.read_csv('Data.csv')\n",
        "X = dataset.iloc[:, :-1].values\n",
        "y = dataset.iloc[:, 3].values\n",
        "\n",
        "# Taking care of missing data\n",
        "# Updated Imputer\n",
        "from sklearn.impute import SimpleImputer\n",
        "missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)\n",
        "missingvalues = missingvalues.fit(X[:, 1:3])\n",
        "X[:, 1:3]=missingvalues.transform(X[:, 1:3])\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V6sUO5mfqt6z",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 187
        },
        "outputId": "71839c80-b620-4887-e81b-a44282b2c6f9"
      },
      "source": [
        "#Colab doesn't have the built in variable explorer, so we can print our variables out. For example:\n",
        "print (X)"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[['France' 44.0 72000.0]\n",
            " ['Spain' 27.0 48000.0]\n",
            " ['Germany' 30.0 54000.0]\n",
            " ['Spain' 38.0 61000.0]\n",
            " ['Germany' 40.0 63777.77777777778]\n",
            " ['France' 35.0 58000.0]\n",
            " ['Spain' 38.77777777777778 52000.0]\n",
            " ['France' 48.0 79000.0]\n",
            " ['Germany' 50.0 83000.0]\n",
            " ['France' 37.0 67000.0]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ae2iFe3Rn2iO",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Data Preprocessing\n",
        "\n",
        "# Importing the libraries\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "\n",
        "# Importing the dataset\n",
        "dataset = pd.read_csv('Data.csv')\n",
        "X = dataset.iloc[:, :-1].values\n",
        "y = dataset.iloc[:, 3].values\n",
        "\n",
        "# Taking care of missing data\n",
        "# Updated Imputer\n",
        "from sklearn.impute import SimpleImputer\n",
        "missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)\n",
        "missingvalues = missingvalues.fit(X[:, 1:3])\n",
        "X[:, 1:3]=missingvalues.transform(X[:, 1:3])\n",
        "\n",
        "\n",
        "# Encoding categorical data\n",
        "# Encoding the Independent Variable\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.preprocessing import LabelEncoder, OneHotEncoder\n",
        "#labelencoder_X = LabelEncoder()\n",
        "#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])\n",
        "#onehotencoder = OneHotEncoder(categorical_features = [0])\n",
        "#X = onehotencoder.fit_transform(X).toarray()\n",
        "# Encoding the Dependent Variable\n",
        "#labelencoder_y = LabelEncoder()\n",
        "#y = labelencoder_y.fit_transform(y)\n",
        "\n",
        "ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')\n",
        "X = np.array(ct.fit_transform(X), dtype=np.float)\n",
        "# Encoding Y data\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "y = LabelEncoder().fit_transform(y)\n",
        "# Splitting the dataset into the Training set and Test set\n",
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)\n",
        "# Feature Scaling\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "sc_X = StandardScaler()\n",
        "X_train = sc_X.fit_transform(X_train)\n",
        "X_test = sc_X.transform(X_test)\n",
        "sc_y = StandardScaler()\n",
        "y_train = sc_y.fit_transform(y_train.reshape(-1,1))"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tPXrnm0Dn2ek",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "9dfafde7-0f58-462c-e896-dddb66ecf28e"
      },
      "source": [
        "#Colab doesn't have the built in variable explorer, so we can print our variables out. For example: \n",
        "print (y)"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[0 1 0 0 1 1 0 1 0 1]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "W6My8YVOpFYl",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "J6epjyw1pFgc",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "spYXA5FApFpd",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0guxKKH0pFnv",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}