using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.IO; // 需要添加 IO 命名空间
using UnityEditor; // 需要添加 UnityEditor 命名空间

public class ModelGenerator : MonoBehaviour
{
    public Button generateButton;
    public Texture2D sourceImage;
    public Vector3 spawnPosition = new Vector3(-9.726f, 1.982f, 75.398f);
    public Vector3 spawnRotationEuler = new Vector3(-90f, -90f, 0f); // 添加旋转控制
    public string modelsRelativePath = "Models"; // TripoSR 输出模型的相对路径 (相对于 Assets)

    private TripoSRForUnity tripoSR;
    private bool isGenerating = false;
    private string expectedModelName = "";

    void Start()
    {
        tripoSR = GetComponent<TripoSRForUnity>();
        if (tripoSR == null)
        {
            tripoSR = gameObject.AddComponent<TripoSRForUnity>();
            Debug.LogWarning("TripoSRForUnity component was not found and has been added. Please configure its Python Path.");
        }

        // 确保 TripoSRForUnity 配置正确
        // 注意：直接修改其他组件的私有变量是不推荐的，这里仅作提示
        // 您应该在 Inspector 中手动设置 TripoSRForUnity 的 moveAndRename=true, autoAddMesh=false
        // if (!tripoSR.moveAndRename) Debug.LogError("请在 TripoSRForUnity 组件中勾选 'Move And Rename'!");
        // if (tripoSR.autoAddMesh) Debug.LogError("请在 TripoSRForUnity 组件中取消勾选 'Auto Add Mesh'!");


        generateButton.onClick.AddListener(StartModelGeneration);
        // 监听 Python 进程结束事件
        TripoSRForUnity.OnPythonProcessEnded += HandlePythonProcessEnd;
    }

    void StartModelGeneration()
    {
        if (sourceImage == null)
        {
            Debug.LogError("请先在 ModelGenerator 组件中设置源图片 (Source Image)！");
            return;
        }
        if (isGenerating)
        {
            Debug.LogWarning("正在生成模型，请稍候...");
            return;
        }

        // 检查 TripoSRForUnity 配置 (运行时检查)
        // if (!tripoSR.moveAndRename || tripoSR.autoAddMesh)
        // {
        //     Debug.LogError("请确保 TripoSRForUnity 组件中 'Move And Rename' 已勾选，'Auto Add Mesh' 已取消勾选！");
        //     return;
        // }


        isGenerating = true;
        generateButton.interactable = false;

        // 记录期望的模型文件名 (不含扩展名)
        expectedModelName = Path.GetFileNameWithoutExtension(AssetDatabase.GetAssetPath(sourceImage));
        Debug.Log($"期望的模型文件名: {expectedModelName}.obj");

        tripoSR.SetInputImage(sourceImage);
        tripoSR.RunTripoSR();
        Debug.Log("TripoSR 进程已启动...");
    }

    void HandlePythonProcessEnd()
    {
        Debug.Log("TripoSR Python 进程已结束。开始加载和放置模型...");
        // 启动协程来处理模型加载和放置，给文件移动和 AssetDatabase 一点时间
        StartCoroutine(LoadAndPlaceModel());
    }

    IEnumerator LoadAndPlaceModel()
    {
        // 构建模型文件的期望路径 (相对于 Assets)
        string assetPath = $"Assets/{modelsRelativePath}/{expectedModelName}.obj";
        string fullPath = Path.Combine(Application.dataPath, modelsRelativePath, $"{expectedModelName}.obj");

        Debug.Log($"检查模型文件是否存在于: {fullPath}");

        // 等待文件出现在目标位置
        float timeout = 10f; // 等待超时时间（秒）
        float timer = 0f;
        while (!File.Exists(fullPath) && timer < timeout)
        {
            yield return new WaitForSeconds(0.2f); // 每 0.2 秒检查一次
            timer += 0.2f;
        }

        if (!File.Exists(fullPath))
        {
            Debug.LogError($"等待超时！未能找到模型文件: {assetPath}");
            isGenerating = false;
            generateButton.interactable = true;
            yield break; // 退出协程
        }

        Debug.Log($"文件已找到: {assetPath}。刷新 AssetDatabase 并加载...");

        // 确保 Unity 编辑器识别到新文件
        AssetDatabase.Refresh();
        // 再次等待一小段时间确保刷新完成
        yield return new WaitForSeconds(0.1f);

        // 加载模型预制体
        GameObject loadedPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);

        if (loadedPrefab != null)
        {
            Debug.Log($"成功加载模型: {loadedPrefab.name}。正在实例化...");
            // 实例化模型
            GameObject instantiatedModel = Instantiate(loadedPrefab);
            instantiatedModel.name = expectedModelName; // 可以重命名实例

            // 设置位置
            instantiatedModel.transform.position = spawnPosition;
            Debug.Log($"模型已放置在: {spawnPosition}");

            // *** 添加：设置旋转 ***
            instantiatedModel.transform.rotation = Quaternion.Euler(spawnRotationEuler);
            Debug.Log($"模型旋转已设置 (欧拉角): {spawnRotationEuler}");

            // （可选）如果需要，在这里添加物理组件并设置 isKinematic
            Rigidbody rb = instantiatedModel.GetComponentInChildren<Rigidbody>(); // 尝试获取子物体的 Rigidbody (如果模型结构复杂)
            if (rb == null) // 如果子物体没有，尝试在父物体添加
            {
                 // 检查是否有 MeshFilter 才添加碰撞体和刚体
                 if(instantiatedModel.GetComponentInChildren<MeshFilter>() != null)
                 {
                    GameObject targetForPhysics = instantiatedModel.transform.childCount > 0 ? instantiatedModel.transform.GetChild(0).gameObject : instantiatedModel;
                    rb = targetForPhysics.AddComponent<Rigidbody>();
                    targetForPhysics.AddComponent<MeshCollider>().convex = true; // 或 false
                 }
            }

            if (rb != null)
            {
                rb.isKinematic = true;
                Debug.Log($"已将 Rigidbody 设置为 isKinematic = true");
            }

        }
        else
        {
            Debug.LogError($"加载模型失败: {assetPath}。请检查路径和文件是否有效。");
        }

        // 恢复状态
        isGenerating = false;
        generateButton.interactable = true;
    }


    void OnDestroy()
    {
        if (generateButton != null)
            generateButton.onClick.RemoveListener(StartModelGeneration);
        // 取消监听 Python 进程结束事件
        TripoSRForUnity.OnPythonProcessEnded -= HandlePythonProcessEnd;
    }
}