BoosterHandle booster;
  const char *model_path = "/path/of/model";

  // create booster handle first
  XGBoosterCreate(NULL, 0, &booster);

  // by default, the seed will be set 0
  XGBoosterSetParam(booster, "seed", "0");

  // load model
  XGBoosterLoadModel(booster, model_path);

  const int feat_size = 100;
  const int num_row = 1;
  float feat[num_row][feat_size];

  // create some fake data for predicting
  for (int i = 0; i < num_row; ++i) {
    for(int j = 0; j < feat_size; ++j) {
      feat[i][j] = (i + 1) * (j + 1)
    }
  }

  // convert 2d array to DMatrix
  DMatrixHandle dtest;
  XGDMatrixCreateFromMat(reinterpret_cast<float*>(feat),
                         num_row, feat_size, NAN, &dtest);

  // predict
  bst_ulong out_len;
  const float *f;
  XGBoosterPredict(booster, dtest, 0, 0, &out_len, &f);
  assert(out_len == num_row);
  std::cout << f[0] << std::endl;

  // free memory
  XGDMatrixFree(dtest);
  XGBoosterFree(booster);
