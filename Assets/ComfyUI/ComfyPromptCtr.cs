using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.IO;

[System.Serializable]
public class ResponseData
{
    public string prompt_id;
}

public class ComfyPromptCtr : MonoBehaviour
{
    [Header("UI References")]
    public InputField pInput, nInput;
    
    [Header("Workflow Settings")]
    [Tooltip("Drag and drop your workflow JSON file here")]
    public TextAsset workflowFile;
    
    private string workflowJson;

    private void Start()
    {
        if (workflowFile != null)
        {
            workflowJson = workflowFile.text;
            Debug.Log("Workflow JSON loaded successfully");
        }
        else
        {
            Debug.LogError("Please assign a workflow JSON file in the Inspector");
        }
    }

    public void QueuePrompt()
    {
        StartCoroutine(QueuePromptCoroutine(pInput.text, nInput.text));
    }

    private IEnumerator QueuePromptCoroutine(string positivePrompt, string negativePrompt)
    {
        string url = "http://127.0.0.1:8188/prompt";
        string promptText = GeneratePromptJson();
        promptText = promptText.Replace("Pprompt", positivePrompt);
        promptText = promptText.Replace("Nprompt", negativePrompt);
        Debug.Log(promptText);

        UnityWebRequest request = new UnityWebRequest(url, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(promptText);
        request.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.Log(request.error);
        }
        else
        {
            Debug.Log("Prompt queued successfully." + request.downloadHandler.text);

            ResponseData data = JsonUtility.FromJson<ResponseData>(request.downloadHandler.text);
            Debug.Log("Prompt ID: " + data.prompt_id);
            GetComponent<ComfyWebsocket>().promptID = data.prompt_id;
        }
    }

    private string GeneratePromptJson()
    {
        string guid = Guid.NewGuid().ToString();
        string promptJsonWithGuid = $@"
{{
    ""id"": ""{guid}"",
    ""prompt"": {workflowJson}
}}";

        return promptJsonWithGuid;
    }
}
