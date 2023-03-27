import os


if __name__ == '__main__':
    required_folders = ['checkpoints', 'dataset', 'logs', 'results']
    for _folder in required_folders:
        if not os.path.exists(_folder):
            os.mkdir(_folder)
            print('Create folder: ./%s' % _folder)
        else:
            print('Folder exists: ./%s' % _folder)
    print('Check folders done.')
