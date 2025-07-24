// using System;
// using System.Collections;
// using UnityEngine;
// using UnityEngine.UI;
// using UnityEngine.Networking;
// using System.IO;

// [System.Serializable]
// public class ResponseData
// {
//     public string prompt_id;
// }

// public class ComfyPromptCtr : MonoBehaviour
// {
//     [Header("UI References")]
//     public InputField pInput;  // 正面提示词输入框
//     public InputField nInput;  // 负面提示词输入框
//     public InputField workflowInput;  // 工作流JSON输入框
//     public Button loadWorkflowButton;  // 加载工作流按钮
//     public Button generateButton;  // 生成按钮

//     [Header("Workflow Settings")]
//     public TextAsset defaultWorkflow;  // 默认工作流JSON文件
//     private string currentWorkflow;  // 当前使用的工作流

//     private void Start()
//     {
//         // 初始化默认工作流
//         if (defaultWorkflow != null)
//         {
//             currentWorkflow = defaultWorkflow.text;
//         }

//         // 添加按钮监听
//         if (loadWorkflowButton != null)
//         {
//             loadWorkflowButton.onClick.AddListener(LoadWorkflowFromFile);
//         }
//     }

//     // 从文件加载工作流
//     public void LoadWorkflowFromFile()
//     {
//         StartCoroutine(LoadWorkflowCoroutine());
//     }

//     private IEnumerator LoadWorkflowCoroutine()
//     {
//         // 使用文件选择器选择JSON文件
//         string path = EditorUtility.OpenFilePanel("Select Workflow JSON", "", "json");
//         if (string.IsNullOrEmpty(path)) yield break;

//         // 读取文件内容
//         string jsonContent = File.ReadAllText(path);
//         if (!string.IsNullOrEmpty(jsonContent))
//         {
//             currentWorkflow = jsonContent;
//             if (workflowInput != null)
//             {
//                 workflowInput.text = jsonContent;
//             }
//             Debug.Log("Workflow loaded successfully");
//         }
//     }

//     // 生成图片
//     public void QueuePrompt()
//     {
//         StartCoroutine(QueuePromptCoroutine());
//     }

//     private IEnumerator QueuePromptCoroutine()
//     {
//         string url = "http://127.0.0.1:8188/prompt";
//         string promptText = GeneratePromptJson();
        
//         // 如果有正反提示词输入框，则替换提示词
//         if (pInput != null && nInput != null)
//         {
//             promptText = promptText.Replace("Pprompt", pInput.text);
//             promptText = promptText.Replace("Nprompt", nInput.text);
//         }

//         Debug.Log("Sending prompt: " + promptText);

//         UnityWebRequest request = new UnityWebRequest(url, "POST");
//         byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(promptText);
//         request.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);
//         request.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
//         request.SetRequestHeader("Content-Type", "application/json");

//         yield return request.SendWebRequest();

//         if (request.result != UnityWebRequest.Result.Success)
//         {
//             Debug.LogError("Error: " + request.error);
//         }
//         else
//         {
//             Debug.Log("Prompt queued successfully: " + request.downloadHandler.text);
//             ResponseData data = JsonUtility.FromJson<ResponseData>(request.downloadHandler.text);
//             Debug.Log("Prompt ID: " + data.prompt_id);
//             GetComponent<ComfyWebsocket>().promptID = data.prompt_id;
//         }
//     }

//     private string GeneratePromptJson()
//     {
//         string guid = Guid.NewGuid().ToString();
//         string workflowJson = workflowInput != null ? workflowInput.text : currentWorkflow;

//         string promptJsonWithGuid = $@"{{
//             ""id"": ""{guid}"",
//             ""prompt"": {workflowJson}
//         }}";

//         return promptJsonWithGuid;
//     }
// }