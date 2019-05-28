## This is intended to be run from Azure Devops as a test of succesfull docker deployment
import json
import numpy as np



def test():
# find 30 random samples from test set
    n = 30
    sample_indices = np.random.permutation(X_test.shape[0])[0:n]

    test_samples = json.dumps({"data": X_test[sample_indices].tolist()})
    test_samples = bytes(test_samples, encoding='utf8')

    # predict using the deployed model
    result = service.run(input_data=test_samples)

    # compare actual value vs. the predicted values:
    i = 0
    plt.figure(figsize = (20, 1))

    for s in sample_indices:
        plt.subplot(1, n, i + 1)
        plt.axhline('')
        plt.axvline('')

        # use different color for misclassified sample
        font_color = 'red' if y_test[s] != result[i] else 'black'
        clr_map = plt.cm.gray if y_test[s] != result[i] else plt.cm.Greys

        plt.text(x=10, y =-10, s=result[i], fontsize=18, color=font_color)
        plt.imshow(X_test[s].reshape(28, 28), cmap=clr_map)

        i = i + 1
    plt.show()
