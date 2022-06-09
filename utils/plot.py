import matplotlib.pyplot as plt

def plot(loss, title, losses):
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("# epoch")
    plt.ylabel(loss)
    plt.title(title)
    plt.savefig('./plots/' + title + '.png')
    plt.close()


def plot_matrix(out, classifier, sample, fold):
    fig = plt.figure()
    plt.pcolor(abs(out))
    plt.colorbar()
    plt.imshow(out)
    title = "Output, Classifier = " + classifier
    plt.title(title)
    plt.savefig('./plots/' + classifier + str(sample) + 'fold' + str(fold) + '.png')
    plt.close()


