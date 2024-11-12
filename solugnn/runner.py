from model_service import ModelService


def main():
    gcn_multi = ModelService(multi_model=True)
    gcn_multi._set_globals()
    gcn_multi.load_models()
    smile = 'CC(=O)N1CCCC1'
    mean, sd = gcn_multi.get_preds(smile)
    print(f'{smile}\nSolvation free energy: {mean:.2f} ± {sd:.3f}')


if __name__ == '__main__':
    main()
